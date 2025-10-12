import os
import tvm
import tvm.meta_schedule as ms
import tvm
import tvm.tir.tensor_intrin.cuda
import tvm.topi as topi
from tvm import te
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
print(f"DEBUG: SCRIPT_DIR={SCRIPT_DIR}")
print(f"DEBUG: PROJECT_ROOT={PROJECT_ROOT}")

import glob
layer_dirs = [os.path.basename(d) for d in glob.glob(os.path.join(PROJECT_ROOT, "layer_*")) if os.path.isdir(d)]
print(f"Found {len(layer_dirs)} layer directories.")

def create_conv2d_module(input_shape, filter_shape, strides, padding, dilation, layout, dtype):
    A = te.placeholder(input_shape, name="A", dtype=dtype)
    W = te.placeholder(filter_shape, name="W", dtype=dtype)
    C = topi.nn.conv2d(A, W, strides, padding, dilation, data_layout=layout, out_dtype=dtype)
    prim_func = te.create_prim_func([A, W, C])
    return tvm.IRModule({"main": prim_func})


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

def extract_features_to_csv(db, target, out_csv_path, layer_dir):
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
        # print(f"Features shape {features.shape}, layer_dir={layer_dir}")
        features_list.append(features.numpy().flatten())
    # Save to CSV
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in features_list:
            writer.writerow(row)
    print(f"Saved features for {len(features_list)} records to {out_csv_path}")

import multiprocessing

def process_layer(layer_dir):
    print(f"Processing directory: {layer_dir}")
    if layer_dir == "layer_0":
        print("skipping layer_0")
        return  # Skip layer_0
    db = load_database(layer_dir)
    import torch
    if "4090" in torch.cuda.get_device_name(0):
        target = tvm.target.Target("nvidia/geforce-rtx-4090")
    elif "3080" in torch.cuda.get_device_name(0):
        target = tvm.target.Target("nvidia/geforce-rtx-3080-ti")
    else:
        raise ValueError("Unsupported GPU architecture")
    out_csv_path = os.path.join(layer_dir, "features.csv")
    extract_features_to_csv(db, target, out_csv_path, layer_dir)

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        pool.map(process_layer, layer_dirs)