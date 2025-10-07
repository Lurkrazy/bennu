
import os
import tvm
import tvm.meta_schedule as ms
from tvm import relax
from tvm.ir import transform
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
import tvm
import tvm.tir.tensor_intrin.cuda
import json
import numpy as np
import time
import tvm.topi as topi
from tvm import te

def get_ms_time(log):
    best_time = [9999]
    with open(log, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            params = data[1]
            time = params[1]
            if np.mean(best_time) > np.mean(time):
                best_time = time
    return best_time


def create_conv2d_module(input_shape, filter_shape, strides, padding, dilation, layout, dtype):
    A = te.placeholder(input_shape, name="A", dtype=dtype)
    W = te.placeholder(filter_shape, name="W", dtype=dtype)
    C = topi.nn.conv2d(A, W, strides, padding, dilation, data_layout=layout, out_dtype=dtype)
    prim_func = te.create_prim_func([A, W, C])
    return tvm.IRModule({"main": prim_func})

def dump_cuda_artifacts(rt_mod, out_dir, stem="kernel"):
    os.makedirs(out_dir, exist_ok=True)
    # 导出 host+device 完整库，便于复现
    try:
        lib_path = os.path.join(out_dir, f"{stem}.so")
        rt_mod.export_library(lib_path)
        print(f"[saved] {lib_path}")
    except Exception as e:
        print(f"[warn] export_library failed: {e}")

    # 抓设备端模块源码/PTX
    try:
        dev_mod = rt_mod.imported_modules[0]
    except Exception as e:
        print(f"[warn] no imported device module: {e}")
        return

    # CUDA C 源
    try:
        cu_src = dev_mod.get_source()  # 常见为 CUDA C（NVRTC 路径）
        if cu_src and len(cu_src) > 0:
            cu_path = os.path.join(out_dir, f"{stem}.cu")
            with open(cu_path, "w") as f:
                f.write(cu_src)
            print(f"[saved] {cu_path}")
    except Exception as e:
        print(f"[warn] get_source() failed: {e}")

    # PTX 源
    try:
        ptx_src = dev_mod.get_source("ptx")
        if ptx_src and len(ptx_src) > 0:
            ptx_path = os.path.join(out_dir, f"{stem}.ptx")
            with open(ptx_path, "w") as f:
                f.write(ptx_src)
            print(f"[saved] {ptx_path}")
    except Exception as e:
        print(f"[warn] get_source('ptx') failed: {e}")
        
    # asm
    try:
        sm_asm = dev_mod.get_source("sass")
        if sm_asm and len(sm_asm) > 0:
            sass_path = os.path.join(out_dir, f"{stem}.sass")
            with open(sass_path, "w") as f:
                f.write(sm_asm)
            print(f"[saved] {sass_path}")
    except Exception as e:
        print(f"[warn] get_source('sass') failed: {e}")
    
    # asm
    try:
        sm_asm = dev_mod.get_source("asm")
        if sm_asm and len(sm_asm) > 0:
            asm_path = os.path.join(out_dir, f"{stem}.asm")
            with open(asm_path, "w") as f:
                f.write(sm_asm)
            print(f"[saved] {asm_path}")
    except Exception as e:
        print(f"[warn] get_source('asm') failed: {e}")

    # cubin (guarded: some Module types don't expose get_binary)
    try:
        if hasattr(dev_mod, "get_binary"):
            cubin = dev_mod.get_binary("cubin")
            if cubin and len(cubin) > 0:
                cubin_path = os.path.join(out_dir, f"{stem}.cubin")
                with open(cubin_path, "wb") as f:
                    f.write(cubin)
                print(f"[saved] {cubin_path}")
        else:
            # no get_binary available on this module; skip writing cubin
            print(f"[warn] device module has no get_binary; skipping cubin export")
    except Exception as e:
        print(f"[warn] get_binary('cubin') failed: {e}")
    


def local_build_and_run(
    mod: IRModule,
    target: tvm.target.Target,
    device: tvm.runtime.device,
    inputs: list,
    number: int = 10,
    repeat: int = 3,
) -> tuple:
    """Build and run the module locally with added diagnostics.

    The function will:
    - Print the module's main param count and try to show per-param shape/dtype if available.
    - Print the provided input shapes/dtypes.
    - Print the built library entry name.
    - Reconcile missing output buffers by synthesizing zero-filled arrays when possible,
      but will surface diagnostics if time_evaluator still fails.
    """
    # Diagnostic: inspect module params
    expected_args = None
    try:
        if isinstance(mod, IRModule) and "main" in mod:
            params = mod["main"].params
            expected_args = len(params)
            print(f"[diag] module 'main' has {expected_args} params")
            for idx, p in enumerate(params):
                try:
                    name = getattr(p, "name_hint", None)
                    tanno = getattr(p, "type_annotation", None)
                    shape = None
                    dtype = None
                    if tanno is not None and hasattr(tanno, "shape"):
                        try:
                            shape = tuple(int(x) for x in tanno.shape)
                        except Exception:
                            # shape may contain PrimExprs; show raw
                            shape = tuple(str(x) for x in tanno.shape)
                        dtype = getattr(tanno, "dtype", None)
                    print(f"[diag] param {idx}: name={name}, shape={shape}, dtype={dtype}")
                except Exception as e:
                    print(f"[diag] param {idx}: failed to inspect param: {e}")
    except Exception as e:
        print(f"[diag] failed to introspect module params: {e}")

    # Show provided inputs
    try:
        for j, inp in enumerate(inputs):
            try:
                print(f"[diag] provided input {j}: shape={getattr(inp,'shape',None)}, dtype={getattr(inp,'dtype',None)}")
            except Exception:
                print(f"[diag] provided input {j}: (unable to read shape/dtype)")
    except Exception:
        pass

    # Reconcile provided inputs vs expected args (best-effort)
    provided = list(inputs)
    if expected_args is not None:
        if len(provided) > expected_args:
            print(f"[warn] more inputs provided ({len(provided)}) than function expects ({expected_args}); trimming extras")
            provided = provided[:expected_args]
        elif len(provided) < expected_args:
            to_add = expected_args - len(provided)
            for idx in range(to_add):
                fallback_shape = None
                fallback_dtype = None
                if len(provided) > 0:
                    fallback_shape = provided[0].shape
                    fallback_dtype = provided[0].dtype
                try:
                    param = mod["main"].params[len(provided) + idx]
                    tanno = getattr(param, "type_annotation", None)
                    if tanno is not None and hasattr(tanno, "shape"):
                        try:
                            shape = tuple(int(x) for x in tanno.shape)
                        except Exception:
                            shape = fallback_shape
                        dtype = getattr(tanno, "dtype", None) or fallback_dtype or "float32"
                    else:
                        shape = fallback_shape
                        dtype = fallback_dtype or "float32"
                except Exception:
                    shape = fallback_shape
                    dtype = fallback_dtype or "float32"

                if shape is None:
                    shape = (1,)
                try:
                    z = np.zeros(shape, dtype=np.dtype(dtype))
                except Exception:
                    z = np.zeros((1,), dtype=np.float32)
                print(f"[info] added zero buffer for missing arg {len(provided)+idx} with shape {z.shape} and dtype {z.dtype}")
                provided.append(z)

    # Build the module and print entry name
    lib = tvm.build(mod, target=target)
    try:
        print(f"[diag] built lib.entry_name = {lib.entry_name}")
    except Exception:
        print("[diag] built lib has no entry_name attribute")

    # Convert to tvm NDArrays on target device
    tvm_inputs = [tvm.nd.array(inp, device=device) for inp in provided]
    device.sync()

    # Attempt to run benchmark and provide verbose diagnostics on failure
    try:
        func = lib.time_evaluator(lib.entry_name, dev=device, number=number, repeat=repeat)
        benchmark_res = func(*tvm_inputs)
        device.sync()
        return [arg.numpy() for arg in tvm_inputs], list(benchmark_res.results)
    except Exception as e:
        provided_n = len(tvm_inputs)
        msg = f"Benchmark failed: {e}. Provided args: {provided_n}"
        if expected_args is not None:
            msg = f"{msg}, expected args: {expected_args}"
        # Add a traceback-like hint
        print(f"[error] {msg}")
        # Re-raise detailed error to surface to caller
        raise RuntimeError(msg) from e


def load_database(work_dir: str) -> ms.database.JSONDatabase:
    """
    Load a JSON-based meta-schedule database from existing files.

    Args:
        work_dir: Directory containing database_workload.json and database_tuning_record.json

    Returns:
        JSONDatabase instance ready for queries

    Note:
        Database files are typically created during tuning runs using:
        - relax.transform.MetaScheduleTuneIRMod
        - relax.transform.MetaScheduleTuneTIR
    """
    workload_path = os.path.join(work_dir, "database_workload.json")
    records_path = os.path.join(work_dir, "database_tuning_record.json")

    # Create database instance pointing to existing JSON files
    db = ms.database.JSONDatabase(workload_path, records_path)

    return db
# Helper: check if CUDA compute supports native fp16 (best-effort)
def have_fp16(compute_version) -> bool:
    try:
        # compute_version may be a tuple like (8,6) or a float/int; try to handle common forms
        if compute_version is None:
            return False
        if isinstance(compute_version, (tuple, list)):
            major, minor = compute_version[0], compute_version[1] if len(compute_version) > 1 else 0
            ver = float(f"{major}.{minor}")
            return ver >= 5.3
        try:
            ver = float(compute_version)
            return ver >= 5.3
        except Exception:
            return True
    except Exception:
        return True


def verify_conv2d(data_dtype, conv_dtype, tensor_format=0, groups=1):
    """
    Reference verification adapted from the provided snippet.

    Uses cuDNN conv_forward schedule on the GPU when available and compares
    the result against TOPI's Python reference (float32) with final casting
    to the requested data_dtype (e.g., float16). Raises on mismatch via
    tvm.testing.assert_allclose.

    Returns True on success, False if the check was skipped (e.g., no fp16 support).
    """
    import tvm.topi.cudnn as cudnn  # import here to avoid hard dependency at module import time
    from tvm import testing

    in_channel = 4
    out_channel = 16
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1
    batch = 3
    height = 32
    width = 32

    # Check CUDA device
    try:
        dev = tvm.cuda(0)
    except Exception:
        raise RuntimeError("CUDA device not available for verify_conv2d")

    # If fp16 requested, ensure device supports it
    if data_dtype == "float16":
        try:
            if not have_fp16(dev.compute_version):
                print("Skip because gpu does not have fp16 support")
                return False
        except Exception:
            pass

    # shapes depending on format
    if tensor_format == 0:
        xshape = [batch, in_channel, height, width]
        wshape = [out_channel, in_channel // groups, filter_h, filter_w]
    else:
        xshape = [batch, height, width, in_channel]
        wshape = [out_channel, filter_h, filter_w, in_channel // groups]

    X = te.placeholder(xshape, name="X", dtype=data_dtype)
    W = te.placeholder(wshape, name="W", dtype=data_dtype)
    Y = cudnn.conv_forward(
        X,
        W,
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        conv_mode=1,
        tensor_format=tensor_format,
        conv_dtype=conv_dtype,
        algo=-1,
        groups=groups,
    )
    yshape = [int(x) for x in Y.shape]
    s = te.create_schedule(Y.op)

    # build and run
    f = tvm.build(s, [X, W, Y], "cuda --host=llvm", name="conv2d")
    x_np = np.random.uniform(-1, 1, xshape).astype(data_dtype)
    w_np = np.random.uniform(-1, 1, wshape).astype(data_dtype)
    y_np = np.zeros(yshape).astype(data_dtype)
    x = tvm.nd.array(x_np, dev)
    w = tvm.nd.array(w_np, dev)
    y = tvm.nd.array(y_np, dev)

    # reference via TOPI python implementations (compute in float32 then cast)
    if tensor_format == 0:
        c_np = tvm.topi.testing.conv2d_nchw_python(x_np.astype(np.float32), w_np.astype(np.float32), stride_h, pad_h, groups=groups)
    else:
        wt = w_np.transpose((1, 2, 3, 0))  # OHWI => HWIO
        c_np = tvm.topi.testing.conv2d_nhwc_python(x_np.astype(np.float32), wt.astype(np.float32), stride_h, pad_h, groups=groups)

    # cast reference to target dtype
    c_np = c_np.astype(data_dtype)

    # run tvm conv and compare
    f(x, w, y)
    out = y.numpy()

    # use tvm.testing.assert_allclose with fp16 tolerances
    tvm.testing.assert_allclose(y.numpy(), c_np, atol=1e-2, rtol=1e-2)
    return True

shapes_mini = [   
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),   # conv1
]

## ------------------ Global ---------------------
layout = "NCHW"
dtype = "float16"

if __name__ == "__main__":
    
    # database path: data/ms
    logfile_dir = "data/ms/layer_mini_0"
    db = load_database(logfile_dir)  # Load the database from the specified
    input
    for i, shape in enumerate(shapes_mini):
        N, C, H, W, K, _, R, S, _, stride, padding, dilation, groups = shape
        input_shape = (N, C, H, W)
        filter_shape = (K, C, R, S)
        strides = (stride, stride)
        paddings = (padding, padding)
        dilations = (dilation, dilation)
        layer_tag = f"layer_mini_{i}"
        stem = f"{layer_tag}"
        
        print("Database loaded.")
        print(f"Workload path: {db.path_workload}")
        print(f"Tuning record path: {db.path_tuning_record}")

        # Example: Query best tuning record for a specific workload
        best_time = get_ms_time(db.path_tuning_record)  

        mean_time = np.mean(best_time) * 1000
        std_time = np.std(best_time) * 1000

        print(f"Best time (ms): {mean_time:.6f}")
        print(f"Best std  (ms): {std_time:.6f}")
        
        # input("Press Enter to continue...")
        
        # # Example: Query all workloads in the database
        # workloads = db.get_all_workloads()
        # print(f"Total workloads in database: {len(workloads)}")
        # for i, wl in enumerate(workloads[:5]):  # Print first 5 workloads
        #     print(f"Workload {i}: {wl}")
        
        # input("Press Enter to continue...")
        target = tvm.target.Target("nvidia/geforce-rtx-3080-ti")
        out_dir = "data/ms/artifacts"
        mod = create_conv2d_module(input_shape, filter_shape, strides, paddings, dilations, layout, dtype)

        # 从 DB 拿最佳 schedule → build → 导出 CUDA 源/ptx/so 并执行 workload 进行简单验证/基准
        try:
            sch = ms.tir_integration.compile_tir(db, mod, target)
            if sch is None:
                print("[warn] No valid schedule from database.")
            else:
                # 可视化 schedule / trace（非必要）
                try:
                    sch.mod.show()
                    sch.trace.show()
                except Exception:
                    pass

                # Build runtime module (host + device)
                try:
                    rt_mod = tvm.build(sch.mod, target=target)  # host+device
                    dump_cuda_artifacts(rt_mod, logfile_dir, stem=stem)
                except Exception as e:
                    print(f"[warn] build/export failed: {e}")
                    rt_mod = None

                # Prepare inputs for the primfunc (A, W, C) based on shapes and run benchmark
                try:
                    # Create random inputs with correct dtype
                    np_dtype = np.dtype(dtype)
                    a_np = np.random.uniform(-1, 1, size=input_shape).astype(np_dtype)
                    w_np = np.random.uniform(-1, 1, size=filter_shape).astype(np_dtype)

                    # For conv2d primfuncs created via te.create_prim_func([A, W, C])
                    # the third argument is the output buffer C with shape (N, K, out_h, out_w).
                    try:
                        N, C_in, H, W_in = input_shape
                        K = filter_shape[0]
                        R = filter_shape[2]
                        S = filter_shape[3]
                        stride = strides[0]
                        padding = paddings[0]
                        dilation = dilations[0]
                        out_h = (H + 2 * padding - dilation * (R - 1) - 1) // stride + 1
                        out_w = (W_in + 2 * padding - dilation * (S - 1) - 1) // stride + 1
                        c_shape = (N, K, out_h, out_w)
                        c_np = np.zeros(c_shape, dtype=np_dtype)
                        inputs_np = [a_np, w_np, c_np]
                    except Exception:
                        # Fallback: pass inputs only (some primfuncs may expect only inputs and return output)
                        inputs_np = [a_np, w_np]

                    # Choose device
                    try:
                        device = tvm.device("cuda", 0)
                    except Exception:
                        raise RuntimeError("No CUDA device available")

                    # Run the compiled scheduled module (use sch.mod to include schedule if available)
                    run_mod = sch.mod if sch is not None else mod
                    try:
                        results, run_secs = local_build_and_run(run_mod, target, device, inputs_np, number=10, repeat=3)
                        mean_ms = np.mean(run_secs) * 1000
                        std_ms = np.std(run_secs) * 1000
                        print(f"Run result: mean {mean_ms:.6f} ms, std {std_ms:.6f} ms")
    
                        # Correctness check using PyTorch fp16 reference; fallback to a numpy fp16-like implementation.
                        try:
                            import torch
                            import torch.nn.functional as F
                            have_torch = True
                        except Exception as torch_err:
                            print(f"[warn] PyTorch import failed (will use numpy fp16 reference): {torch_err}")
                            have_torch = False
    
                        try:
                            # Determine scheduled output buffer (prefer 3rd arg)
                            sched_out = None
                            if isinstance(results, (list, tuple)) and len(results) >= 3:
                                sched_out = results[2]
                            elif isinstance(results, (list, tuple)) and len(results) >= 1:
                                sched_out = results[-1]
    
                            if sched_out is None:
                                print("[warn] no scheduled output available for correctness check")
                            else:
                                # Convert scheduled output to float16 numpy if possible
                                try:
                                    sched_fp16 = sched_out.astype(np.float16)
                                except Exception:
                                    sched_fp16 = sched_out.astype(np.float32).astype(np.float16)
    
                                # Compute reference in fp16 when possible
                                if have_torch:
                                    try:
                                        # Prefer GPU fp16 if available
                                        use_cuda = torch.cuda.is_available()
                                        device_torch = torch.device("cuda" if use_cuda else "cpu")
    
                                        t_input = torch.from_numpy(a_np.astype(np.float16)).to(device_torch)
                                        t_weight = torch.from_numpy(w_np.astype(np.float16)).to(device_torch)
    
                                        with torch.no_grad():
                                            # Perform conv in fp16
                                            ref_t = F.conv2d(
                                                t_input,
                                                t_weight,
                                                bias=None,
                                                stride=strides,
                                                padding=paddings,
                                                dilation=dilations,
                                                groups=groups,
                                            )
                                        ref_np = ref_t.cpu().numpy().astype(np.float16)
                                    except Exception as e_torch_fp16:
                                        # Fallback: compute in fp32 then cast to fp16 with warning
                                        print(f"[warn] PyTorch fp16 conv failed, falling back to fp32->fp16: {e_torch_fp16}")
                                        t_input = torch.from_numpy(a_np.astype(np.float32))
                                        t_weight = torch.from_numpy(w_np.astype(np.float32))
                                        with torch.no_grad():
                                            ref = F.conv2d(
                                                t_input,
                                                t_weight,
                                                bias=None,
                                                stride=strides,
                                                padding=paddings,
                                                dilation=dilations,
                                                groups=groups,
                                            )
                                        ref_np = ref.cpu().numpy().astype(np.float16)
                                else:
                                    # Numpy fp16-like convolution: accumulate in float16 to mimic fp16 behavior
                                    def conv2d_numpy_fp16(inp, w, stride, padding, dilation, groups):
                                        # inp: N,C,H,W  w: K,C, R,S
                                        inp_h = inp.astype(np.float16)
                                        w_h = w.astype(np.float16)
                                        N, C_in, H_in, W_in = inp_h.shape
                                        K, Cw, R, S = w_h.shape
                                        assert C_in == Cw or Cw * groups == C_in, "channel mismatch"
                                        out_h = (H_in + 2 * padding - dilation * (R - 1) - 1) // stride + 1
                                        out_w = (W_in + 2 * padding - dilation * (S - 1) - 1) // stride + 1
                                        out = np.zeros((N, K, out_h, out_w), dtype=np.float16)
                                        pad_top = pad_bottom = padding
                                        pad_left = pad_right = padding
                                        inp_padded = np.pad(inp_h, ((0,0),(0,0),(pad_top,pad_bottom),(pad_left,pad_right)), mode="constant", constant_values=0).astype(np.float16)
                                        for n in range(N):
                                            for k in range(K):
                                                for oh in range(out_h):
                                                    for ow in range(out_w):
                                                        acc = np.float16(0.0)
                                                        for c in range(C_in):
                                                            for r in range(R):
                                                                for s in range(S):
                                                                    ih = oh * stride + r * dilation
                                                                    iw = ow * stride + s * dilation
                                                                    prod = np.float16(inp_padded[n, c, ih, iw] * w_h[k, c, r, s])
                                                                    acc = np.float16(acc + prod)
                                                        out[n, k, oh, ow] = acc
                                        return out
                                    ref_np = conv2d_numpy_fp16(a_np, w_np, stride=strides[0], padding=paddings[0], dilation=dilations[0], groups=groups)
    
                                # Prepare arrays for comparison: bring both to float32 for numeric diagnostics but computed as fp16
                                ref_cmp = ref_np.astype(np.float32)
                                sched_cmp = sched_fp16.astype(np.float32)
    
                                # Align shapes if necessary
                                if sched_cmp.shape != ref_cmp.shape:
                                    try:
                                        sched_cmp = sched_cmp.reshape(ref_cmp.shape)
                                    except Exception:
                                        pass
    
                                abs_diff = np.abs(sched_cmp - ref_cmp)
                                max_diff = float(np.max(abs_diff))
                                mean_diff = float(np.mean(abs_diff))
                                rel_max = max_diff / (np.max(np.abs(ref_cmp)) + 1e-8)
    
                                # Use fp16-specific tolerances (atol / rtol).
                                # Typical choices: 1e-2 or 1e-3; default to 1e-2 here.
                                atol_fp16 = 1e-2
                                rtol_fp16 = 1e-2
                                # pass when arrays are elementwise close under fp16 tolerances
                                pass_condition = np.allclose(ref_cmp, sched_cmp, rtol=rtol_fp16, atol=atol_fp16)
    
                                print(f"[check] ref (fp16) shape: {ref_cmp.shape}, sched shape: {sched_cmp.shape}")
                                print(f"[check] max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}, rel_max={rel_max:.6e}")
                                print(f"[check] tolerances: atol={atol_fp16}, rtol={rtol_fp16}")
                                print(f"[check] correctness (vs fp16 reference): {'PASS' if pass_condition else 'FAIL'}")
                        except Exception as e:
                            print(f"[warn] reference check failed: {e}")
                    except Exception as e:
                        print(f"[warn] running benchmark failed: {e}")
                except Exception as e:
                    print(f"[warn] prepare/run inputs failed: {e}")
        except Exception as e:
            print(f"[warn] compile_tir/build/export failed: {e}")

        print(f"Layer {i}: {shape} done.")
        print("-" * 50)
        print(f"Mean time (ms): {mean_time:.6f}, Std time (ms): {std_time:.6f}")