import os
import argparse
import json
import numpy as np
import tvm
from tvm import te, topi
from tvm import meta_schedule as ms
import tvm.tir.tensor_intrin.cuda


def dump_cuda_artifacts(rt_mod, out_dir, stem="kernel"):
    """Dump CUDA artifacts including .so, .cu, .ptx, .sass, .asm, .cubin"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Export host+device complete library
    try:
        lib_path = os.path.join(out_dir, f"{stem}.so")
        rt_mod.export_library(lib_path)
        print(f"[saved] {lib_path}")
    except Exception as e:
        print(f"[warn] export_library failed: {e}")

    # Get device module source code/PTX
    try:
        dev_mod = rt_mod.imported_modules[0]
    except Exception as e:
        print(f"[warn] no imported device module: {e}")
        return

    # CUDA C source
    try:
        cu_src = dev_mod.get_source()
        if cu_src and len(cu_src) > 0:
            cu_path = os.path.join(out_dir, f"{stem}.cu")
            with open(cu_path, "w") as f:
                f.write(cu_src)
            print(f"[saved] {cu_path}")
    except Exception as e:
        print(f"[warn] get_source() failed: {e}")

    # PTX source
    try:
        ptx_src = dev_mod.get_source("ptx")
        if ptx_src and len(ptx_src) > 0:
            ptx_path = os.path.join(out_dir, f"{stem}.ptx")
            with open(ptx_path, "w") as f:
                f.write(ptx_src)
            print(f"[saved] {ptx_path}")
    except Exception as e:
        print(f"[warn] get_source('ptx') failed: {e}")
        
    # SASS
    try:
        sm_asm = dev_mod.get_source("sass")
        if sm_asm and len(sm_asm) > 0:
            sass_path = os.path.join(out_dir, f"{stem}.sass")
            with open(sass_path, "w") as f:
                f.write(sm_asm)
            print(f"[saved] {sass_path}")
    except Exception as e:
        print(f"[warn] get_source('sass') failed: {e}")
    
    # ASM
    try:
        sm_asm = dev_mod.get_source("asm")
        if sm_asm and len(sm_asm) > 0:
            asm_path = os.path.join(out_dir, f"{stem}.asm")
            with open(asm_path, "w") as f:
                f.write(sm_asm)
            print(f"[saved] {asm_path}")
    except Exception as e:
        print(f"[warn] get_source('asm') failed: {e}")

    # CUBIN
    try:
        cubin = dev_mod.get_binary("cubin")
        if cubin and len(cubin) > 0:
            cubin_path = os.path.join(out_dir, f"{stem}.cubin")
            with open(cubin_path, "wb") as f:
                f.write(cubin)
            print(f"[saved] {cubin_path}")
    except Exception as e:
        print(f"[warn] get_binary('cubin') failed: {e}")


def load_database(work_dir: str) -> ms.database.JSONDatabase:
    """
    Load a JSON-based meta-schedule database from existing files.
    
    Args:
        work_dir: Directory containing database_workload.json and database_tuning_record.json
    
    Returns:
        JSONDatabase instance ready for queries
    """
    workload_path = os.path.join(work_dir, "database_workload.json")
    records_path = os.path.join(work_dir, "database_tuning_record.json")
    
    if not os.path.exists(workload_path):
        raise FileNotFoundError(f"Workload database not found: {workload_path}")
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"Tuning records not found: {records_path}")
    
    # Create database instance pointing to existing JSON files
    db = ms.database.JSONDatabase(workload_path, records_path)
    
    return db


def get_ms_time(log):
    """Get best execution time from tuning records"""
    best_time = [9999]
    with open(log, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            params = data[1]
            time_arr = params[1]
            if np.mean(best_time) > np.mean(time_arr):
                best_time = time_arr
    return best_time


def create_matmul_module(M, N, K, dtype):
    """Create a matmul TVM IRModule"""
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    C = topi.nn.matmul(A, B, out_dtype=dtype)
    prim_func = te.create_prim_func([A, B, C])
    return tvm.IRModule({"main": prim_func})


def parse_matmul_shape_from_log(log_dir):
    """Try to infer matmul shape from log directory name or database"""
    # Try to parse from directory name like "matmul_0" or "matmul_4096x4096x4096"
    basename = os.path.basename(log_dir.rstrip('/'))
    
    # Check if there's shape info in the name
    if 'x' in basename:
        parts = basename.split('_')
        for part in parts:
            if 'x' in part:
                try:
                    dims = [int(d) for d in part.split('x')]
                    if len(dims) == 3:
                        return tuple(dims)
                except ValueError:
                    pass
    
    # Default to a common shape if can't parse
    print(f"[warn] Could not parse shape from directory name, using default 4096x4096x4096")
    return (4096, 4096, 4096)


def main():
    parser = argparse.ArgumentParser(
        description="Dump CUDA artifacts from TVM meta-schedule database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump from a single log directory
  python dump.py -d logs/matmul_0 -t nvidia/geforce-rtx-3080-ti
  
  # Dump with custom shape
  python dump.py -d logs/matmul_0 -t nvidia/geforce-rtx-4090 -s 4096 4096 4096
  
  # Dump with custom output directory and stem name
  python dump.py -d logs/matmul_0 -o artifacts -n matmul_4k
        """
    )
    parser.add_argument("-d", "--log-dir", type=str, required=True,
                        help="Path to meta-schedule log directory (containing database_*.json)")
    parser.add_argument("-t", "--target", type=str, default="nvidia/geforce-rtx-3080-ti",
                        help="TVM target string (e.g., 'nvidia/geforce-rtx-4090', 'nvidia/a100')")
    parser.add_argument("-s", "--shape", type=int, nargs=3, metavar=("M", "N", "K"),
                        help="Matmul shape M N K (e.g., 4096 4096 4096)")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Output directory for artifacts (default: <log-dir>/artifacts)")
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="Stem name for output files (default: inferred from shape)")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Data type for matmul (default: float16)")
    
    args = parser.parse_args()
    
    log_dir = args.log_dir
    target_str = args.target
    dtype = args.dtype
    
    # Validate log directory
    if not os.path.isdir(log_dir):
        print(f"[error] Log directory does not exist: {log_dir}")
        return 1
    
    # Get matmul shape
    if args.shape:
        M, N, K = args.shape
    else:
        M, N, K = parse_matmul_shape_from_log(log_dir)
    
    print(f"[info] Matmul shape: {M}x{N}x{K}, dtype: {dtype}")
    
    # Set output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(log_dir, "artifacts")
    
    # Set stem name
    if args.name:
        stem = args.name
    else:
        stem = f"matmul_{M}x{N}x{K}"
    
    print(f"[info] Output directory: {out_dir}")
    print(f"[info] Stem name: {stem}")
    print(f"[info] Target: {target_str}")
    
    # Load database
    try:
        print(f"\n[Loading database from {log_dir}]")
        db = load_database(log_dir)
        print(f"[info] Database loaded successfully")
        
        # Get best time
        records_path = os.path.join(log_dir, "database_tuning_record.json")
        best_time = get_ms_time(records_path)
        mean_time = np.mean(best_time) * 1000
        std_time = np.std(best_time) * 1000
        print(f"[info] Best time: {mean_time:.6f} ms (±{std_time:.6f} ms)")
        
    except Exception as e:
        print(f"[error] Failed to load database: {e}")
        return 1
    
    # Create matmul module
    try:
        print(f"\n[Creating matmul module]")
        mod = create_matmul_module(M, N, K, dtype)
        print(f"[info] Module created successfully")
    except Exception as e:
        print(f"[error] Failed to create module: {e}")
        return 1
    
    # Compile and dump artifacts
    try:
        print(f"\n[Compiling and dumping artifacts]")
        target = tvm.target.Target(target_str)
        
        # Get best schedule from database
        sch = ms.tir_integration.compile_tir(db, mod, target)
        
        if sch is None:
            print("[error] No valid schedule found in database")
            return 1
        
        print(f"[info] Schedule compiled successfully")
        
        # Build runtime module
        rt_mod = tvm.build(sch.mod, target=target)
        print(f"[info] Runtime module built successfully")
        
        # Dump artifacts
        dump_cuda_artifacts(rt_mod, out_dir, stem=stem)
        
        print(f"\n[success] All artifacts dumped to {out_dir}/")
        
    except Exception as e:
        print(f"[error] Failed to compile or dump: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
