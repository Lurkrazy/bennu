import os, sys, time, argparse, tvm, datetime, csv, shutil
import numpy as np
from tvm import te, topi
from tvm import meta_schedule as ms
from tvm.meta_schedule.runner.config import EvaluatorConfig
from tvm.script import tir as T

num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads * 2 // 3)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads * 2 // 3)
os.environ["OMP_NUM_THREADS"] = str(num_threads * 2 // 3)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *

## ------------------ ResNet-18 Shapes ---------------------
# (batch, C, H, W, K, _, R, S, _, stride, padding, dilation, groups)
shapes_b1 = [
    # resnet-18
    # (batch, C, H, W, K, _, R, S, _, stride, padding, dilation, groups)
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]

## ------------------ Global ---------------------
layout = "NCHW"
dtype = "float16"

## ----------------- Benchmark -------------------
def create_conv2d_module(input_shape, filter_shape, strides, padding, dilation, layout, dtype):
    A = te.placeholder(input_shape, name="A", dtype=dtype)
    W = te.placeholder(filter_shape, name="W", dtype=dtype)
    C = topi.nn.conv2d(
        A, W, strides, padding, dilation, data_layout=layout, out_dtype=dtype
    )
    prim_func = te.create_prim_func([A, W, C])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python mm.py -a x86 -l 'results/ms/cpu_matmul'")
    parser.add_argument(
        "-a", "--arch", type=str, default="cuda", help="Options: x86, aarch64, cuda"
    )
    parser.add_argument("-l", "--logfile_parent", type=str, default=".")
    parser.add_argument("-t", "--trials", type=int, default=1000)
    args = parser.parse_args()

    arch = args.arch
    logfile_parent = args.logfile_parent
    trials = args.trials

    # Create results directory if not exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create CSV file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"conv2d_results_{timestamp}.csv")
    
    # Write CSV header
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Layer', 'N', 'C', 'H', 'W', 'K', 'R', 'S', 'stride', 'padding', 'dilation',
            'Trials', 'Best Time (ms)', 'Std Dev (ms)', 'Tuning Time (min)', 'Status'
        ])

        if arch == "cuda":
            target_name = "cuda"
            target = tvm.target.Target(
                "nvidia/geforce-rtx-3080-ti"
            )
            dev = tvm.cuda()
        else:
            print("Archtecture doesn't support.")
            exit(0)
        
        for i, shape in enumerate(shapes_b1):
            N, C, H, W, K, _, R, S, _, stride, padding, dilation, groups = shape
            input_shape = (N, C, H, W)
            filter_shape = (K, C, R, S)
            strides = (stride, stride)
            paddings = (padding, padding)
            dilations = (dilation, dilation)

            logfile = os.path.join(logfile_parent, f"layer_{i}")
            
            # clean the files
            if os.path.exists(logfile):
                shutil.rmtree(logfile)

            try:
                mod = create_conv2d_module(input_shape, filter_shape, strides, paddings, dilations, layout, dtype)
                mean_time, std_time, tuning_time = ms_execute(mod, logfile, target, target_name, trials)
                status = "Success"
            except Exception as e:
                print(f"An error occurred during benchmark for layer {i}: {e}")
                mean_time, std_time, tuning_time = -1, -1, -1
                status = "Failed"

            writer.writerow([
                i, N, C, H, W, K, R, S, stride, padding, dilation,
                trials, mean_time, std_time, tuning_time, status
            ])
            csvfile.flush()
