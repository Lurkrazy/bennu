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
    
    # # Export host+device complete library
    # try:
    #     lib_path = os.path.join(out_dir, f"{stem}.so")
    #     rt_mod.export_library(lib_path)
    #     print(f"[saved] {lib_path}")
    # except Exception as e:
    #     print(f"[warn] export_library failed: {e}")

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

def create_matmul_module(M, N, K, dtype):
    """Create a matmul TVM IRModule"""
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    C = topi.nn.matmul(A, B, out_dtype=dtype)
    prim_func = te.create_prim_func([A, B, C])
    return tvm.IRModule({"main": prim_func})

def main():
    parser = argparse.ArgumentParser(
        description="Dump CUDA artifacts from TVM meta-schedule database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump from a single log directory
  python dump.py -d logs/matmul_0 -t nvidia/geforce-rtx-3080-ti
  
  # Dump with custom output directory and stem name
  python dump.py -d logs/matmul_0 -o artifacts -n matmul_4k
        """
    )
    parser.add_argument("-d", "--log-dir", type=str, required=True,
                        help="Path to meta-schedule log directory (containing database_*.json)")
    parser.add_argument("-t", "--target", type=str, default="nvidia/geforce-rtx-3080-ti",
                        help="TVM target string (e.g., 'nvidia/geforce-rtx-4090', 'nvidia/a100')")
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
    
    M, N, K = 4096, 4096, 4096
    
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
        
    except Exception as e:
        print(f"[error] Failed to load database: {e}")
        return 1
    
    # Create matmul module
    try:
        print(f"\n[Creating matmul module]")
        mod = create_matmul_module(M, N, K, dtype)
        print(tvm.lower(mod, simple_mode=True))
        # input("Press Enter to continue...")
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
        print("\n[Device module source code:]")
        print(rt_mod.imported_modules[0].get_source())
        # input("Press Enter to continue...")
        print(f"sch.mod: {sch.mod}")
        print("\n[Lowered TIR:]")
        print(tvm.lower(sch.mod, simple_mode=True))
        
        # input("Press Enter to continue...")
        
        print(f"[info] Runtime module built successfully")
        
        # Dump artifacts
        dump_cuda_artifacts(rt_mod, out_dir, stem=stem)
        
        # rerun it to show success message
        
        dev = tvm.cuda(0)
        a = tvm.nd.array(np.random.uniform(size=(M, K)).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(K, N)).astype(dtype), dev)
        c = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
        rt_mod(a, b, c)
        evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=3)
        print("gemm with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))
        
        print(f"\n[success] All artifacts dumped to {out_dir}/")
        
    except Exception as e:
        print(f"[error] Failed to compile or dump: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
