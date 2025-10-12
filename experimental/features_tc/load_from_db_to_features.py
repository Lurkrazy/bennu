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
    # Export the complete host+device library for reproduction
    try:
        lib_path = os.path.join(out_dir, f"{stem}.so")
        rt_mod.export_library(lib_path)
        print(f"[saved] {lib_path}")
    except Exception as e:
        print(f"[warn] export_library failed: {e}")

    # Capture device-side module source/PTX
    try:
        dev_mod = rt_mod.imported_modules[0]
    except Exception as e:
        print(f"[warn] no imported device module: {e}")
        return

    # CUDA C source (commonly NVRTC path)
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
        
    # Assembly (sass)
    try:
        sm_asm = dev_mod.get_source("sass")
        if sm_asm and len(sm_asm) > 0:
            sass_path = os.path.join(out_dir, f"{stem}.sass")
            with open(sass_path, "w") as f:
                f.write(sm_asm)
            print(f"[saved] {sass_path}")
    except Exception as e:
        print(f"[warn] get_source('sass') failed: {e}")
    
    # Assembly (asm)
    try:
        sm_asm = dev_mod.get_source("asm")
        if sm_asm and len(sm_asm) > 0:
            asm_path = os.path.join(out_dir, f"{stem}.asm")
            with open(asm_path, "w") as f:
                f.write(sm_asm)
            print(f"[saved] {asm_path}")
    except Exception as e:
        print(f"[warn] get_source('asm') failed: {e}")

    # Binary (cubin)
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

        # Get the best schedule from the database → build → export CUDA source/PTX/so
        try:
            sch = ms.tir_integration.compile_tir(db, mod, target)
            if sch is None:
                print("[warn] No valid schedule from database.")
            else:
                # Visualization
                try:
                    sch.mod.show()
                    sch.trace.show()
                except Exception:
                    pass
                rt_mod = tvm.build(sch.mod, target=target)  # host+device
                dump_cuda_artifacts(rt_mod, logfile_dir, stem=stem)
        except Exception as e:
            print(f"[warn] compile_tir/build/export failed: {e}")

        print(f"Layer {i}: {shape} done.")
        print("-" * 50)
        print(f"Mean time (ms): {mean_time:.6f}, Std time (ms): {std_time:.6f}")
import csv

def extract_features_to_csv(db, target, out_csv_path):
    """
    Save all tuning record features to CSV
    """
    extractor = ms.feature_extractor.PerStoreFeature()
    tune_ctx = ms.TuneContext(target=target)
    records = db.get_all_tuning_records()
    features_list = []
    for record in records:
        sch = tvm.tir.Schedule(record.workload.mod)
        record.trace.apply_to_schedule(sch, remove_postproc=False)
        candidate = ms.MeasureCandidate(sch=sch, args_info=[])
        (features,) = extractor.extract_from(tune_ctx, candidates=[candidate])
        print("Features shape:", features.shape)
        features_list.append(features.numpy().flatten())
    # Save to CSV
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in features_list:
            writer.writerow(row)
    print(f"Saved features for {len(features_list)} records to {out_csv_path}")

if __name__ == "__main__":
    logfile_dir = "data/ms/layer_mini_0"
    db = load_database(logfile_dir)
    target = tvm.target.Target("nvidia/geforce-rtx-3080-ti")
    out_csv_path = os.path.join(logfile_dir, "features.csv")
    extract_features_to_csv(db, target, out_csv_path)