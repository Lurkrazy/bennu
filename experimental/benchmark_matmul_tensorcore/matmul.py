import os, sys, time, argparse, tvm, datetime, csv, shutil
import numpy as np
from tvm import te, topi
from tvm import meta_schedule as ms
from tvm.meta_schedule.runner.config import EvaluatorConfig
from tvm.script import tir as T
import torch
import json

def get_tvm_target():
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_map = {
        "4090": "nvidia/geforce-rtx-4090",
        "3080": "nvidia/geforce-rtx-3080-ti",
        "a100": "nvidia/a100",
    }
    for key, val in gpu_map.items():
        if key in gpu_name:
            return tvm.target.Target(val)
    raise RuntimeError("Unsupported GPU: " + gpu_name)

def get_ms_time(log):
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
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    C = topi.nn.matmul(A, B, out_dtype=dtype)
    prim_func = te.create_prim_func([A, B, C])
    return tvm.IRModule({"main": prim_func})

def ms_execute(mod, logfile, target, target_name, trials):
    start = time.time()
    database = ms.tune_tir(
        mod=mod,
        target=target,
        max_trials_global=trials,
        num_trials_per_iter=64,
        work_dir=logfile,
        runner=ms.runner.LocalRunner(
            evaluator_config=EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=100,
                enable_cpu_cache_flush=True if target_name == "llvm" else False,
            )
        ),
        cost_model=ms.cost_model.XGBModel(
            extractor=ms.feature_extractor.PerStoreFeature(),
            adaptive_training=False,
        ),
        strategy=ms.search_strategy.EvolutionarySearch(),
    )
    end = time.time()

    best_time = get_ms_time(logfile + "/database_tuning_record.json")
    if not best_time:
        return 0, 0, (end-start)/60

    mean_time = np.mean(best_time) * 1000
    std_time = np.std(best_time) * 1000
    tuning_time = (end - start) / 60

    print(f"Best time (ms): {mean_time:.10f}")
    print(f"Best std  (ms): {std_time:.10f}")
    print(f"Tuning Time (min): {tuning_time:.2f}")

    return mean_time, std_time, tuning_time

def get_tensorcore_gflops(M, N, K, dtype):
    # 4090 FP16 theoretical peak: 2 x 16384 x 1e9 / 1e9 = 32768 GFLOPS
    # A100 FP16 theoretical peak: 2 x 312 x 1e9 / 1e9 = 1248 GFLOPS
    # For demo, use 4090 FP16 peak
    # You may want to detect GPU and set accordingly
    return 32768

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python matmul.py -a cuda -l 'results/ms/matmul'")
    parser.add_argument("-a", "--arch", type=str, default="cuda", help="Options: x86, aarch64, cuda")
    parser.add_argument("-l", "--logfile_parent", type=str, default=".")
    parser.add_argument("-t", "--trials", type=int, default=1000)
    args = parser.parse_args()

    arch = args.arch
    logfile_parent = args.logfile_parent
    trials = args.trials

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"matmul_results_{timestamp}.csv")

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'M', 'N', 'K', 'Trials', 'Best Time (ms)', 'Std Dev (ms)', 'Tuning Time (min)', 'GFLOPS', 'TensorCore Peak', 'Percent of Peak', 'Status'
        ])

        if arch == "cuda":
            target_name = "cuda"
            target = get_tvm_target()
            dev = tvm.cuda()
        else:
            print("Archtecture doesn't support.")
            exit(0)

        # Example matmul shapes
        shapes = [
            # (1024, 1024, 1024),
            # (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]
        dtype = "float16"

        for i, (M, N, K) in enumerate(shapes):
            logfile = os.path.join(logfile_parent, f"matmul_{i}")
            if os.path.exists(logfile):
                shutil.rmtree(logfile)

            try:
                mod = create_matmul_module(M, N, K, dtype)
                mean_time, std_time, tuning_time = ms_execute(mod, logfile, target, target_name, trials)
                gflops = (2 * M * N * K) / (mean_time * 1e6) if mean_time > 0 else 0
                tensorcore_peak = get_tensorcore_gflops(M, N, K, dtype)
                percent_of_peak = (gflops / tensorcore_peak) * 100 if tensorcore_peak > 0 else 0
                status = "Success"
            except Exception as e:
                print(f"An error occurred during benchmark for matmul {i}: {e}")
                mean_time, std_time, tuning_time, gflops, percent_of_peak = -1, -1, -1, -1, -1
                tensorcore_peak = get_tensorcore_gflops(M, N, K, dtype)
                status = "Failed"

            writer.writerow([
                M, N, K, trials, mean_time, std_time, tuning_time, gflops, tensorcore_peak, percent_of_peak, status
            ])
            csvfile.flush()