import os
import tvm
import tvm.meta_schedule as ms
import tvm
import tvm.tir.tensor_intrin.cuda
import tvm.topi as topi
from tvm import te
import csv

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
    # Prepare the target and module for compilation
    import torch
    if "4090" in torch.cuda.get_device_name(0):
        target = tvm.target.Target("nvidia/geforce-rtx-4090")
    elif "3080" in torch.cuda.get_device_name(0):
        target = tvm.target.Target("nvidia/geforce-rtx-3080-ti")
    else:
        raise ValueError("Unsupported GPU architecture")
    out_csv_path = os.path.join(logfile_dir, "features.csv")
    extract_features_to_csv(db, target, out_csv_path)