
import os
import tvm
import tvm.meta_schedule as ms
from tvm import relax
from tvm.ir import transform
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext

# Example target (adjust based on your hardware)
target = tvm.target.Target("llvm --num-cores=16")


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


if __name__ == "__main__":
    
    # database path: data/ms
    db = load_database("data/ms/layer_mini_0")
    print("Database loaded.")
    print(f"Workload path: {db.path_workload}")
    print(f"Tuning record path: {db.path_tuning_record}")

    # Example: Query best tuning record for a specific workload
    best_time = get_ms_time(db.path_tuning_record)  

    mean_time = np.mean(best_time) * 1000
    std_time = np.std(best_time) * 1000

    print(f"Best time (ms): {mean_time:.6f}")
    print(f"Best std  (ms): {std_time:.6f}")
    print(f"Tuning Time (min): {tuning_time:.2f}")
    
    input("Press Enter to continue...")
    
    # Example: Query all workloads in the database
    workloads = db.get_all_workloads()
    print(f"Total workloads in database: {len(workloads)}")
    for i, wl in enumerate(workloads[:5]):  # Print first 5 workloads
        print(f"Workload {i}: {wl}")
        if path_workload is None:
            if work_dir is None:
                raise ValueError("Either path_workload or work_dir must be provided.")
            path_workload = osp.join(work_dir, "database_workload.json")
        if path_tuning_record is None:
            if work_dir is None:
                raise ValueError("Either path_tuning_record or work_dir must be provided.")
            path_tuning_record = osp.join(work_dir, "database_tuning_record.json")