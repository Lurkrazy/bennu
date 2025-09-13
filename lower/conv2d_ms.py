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

from src.utils import *  # 里边提供 get_ms_time

## ------------------ ResNet-18 Shapes ---------------------
# (batch, C, H, W, K, _, R, S, _, stride, padding, dilation, groups)
shapes_b1 = [
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),   # conv1
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),   # conv2
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),   # conv3
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),# conv6
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),# conv7
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),# conv8
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),# conv9
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),# conv10
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),# conv11
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12
]

## ------------------ Minimal Unit Shapes ---------------------
# 尽量小、编译/调优很快；用于打通从 TIR→build→导出源码/ptx 的链路
# 这里给两个示例：1x1 卷积（极小）和 3x3 小图（更接近实际）
shapes_mini = [   
    (1, 1, 8, 8, 1, 1, 1, 1, 1, 1, 0, 1, 1),   # 1x1 mini conv
    (1, 1, 16, 16, 1, 1, 3, 3, 1, 1, 1, 1, 1), # 3x3 mini conv
]

## ------------------ Global ---------------------
layout = "NCHW"
dtype = "float16"

## ----------------- Build helpers -------------------
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

## ----------------- MetaSchedule -------------------
def ms_execute(mod, logfile, target, target_name, trials, export_artifacts=True, stem="kernel"):
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

    # 统计最优时间
    best_time = get_ms_time(os.path.join(logfile, "database_tuning_record.json"))
    if not best_time:
        return 0, 0, (end - start) / 60

    mean_time = np.mean(best_time) * 1000
    std_time = np.std(best_time) * 1000
    tuning_time = (end - start) / 60

    print(f"Best time (ms): {mean_time:.6f}")
    print(f"Best std  (ms): {std_time:.6f}")
    print(f"Tuning Time (min): {tuning_time:.2f}")

    # 从 DB 拿最佳 schedule → build → 导出 CUDA 源/ptx/so
    try:
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("[warn] No valid schedule from database.")
        else:
            # 可视化
            try:
                sch.mod.show()
                sch.trace.show()
            except Exception:
                pass
            rt_mod = tvm.build(sch.mod, target=target)  # host+device
            if export_artifacts:
                dump_cuda_artifacts(rt_mod, logfile, stem=stem)
    except Exception as e:
        print(f"[warn] compile_tir/build/export failed: {e}")

    return mean_time, std_time, tuning_time

## ----------------- Main -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("python conv2d_ms.py -a cuda -l results/ms -t 200 --mini")
    parser.add_argument("-a", "--arch", type=str, default="cuda", help="Options: x86, aarch64, cuda")
    parser.add_argument("-l", "--logfile_parent", type=str, default=".")
    parser.add_argument("-t", "--trials", type=int, default=1000)
    parser.add_argument("--mini", action="store_true", help="Run minimal execution unit only")
    args = parser.parse_args()

    arch = args.arch
    logfile_parent = args.logfile_parent
    trials = args.trials
    run_mini = args.mini

    # results 目录
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # CSV 文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"conv2d_results_{timestamp}.csv")

    if arch != "cuda":
        print("Architecture doesn't support (this script currently targets CUDA).")
        sys.exit(0)

    target_name = "cuda"
    # 你原来的 3080Ti target；如需更显式，也可用 "cuda -arch=sm_86"
    target = tvm.target.Target("nvidia/geforce-rtx-3080-ti")
    dev = tvm.cuda()

    # 选择 shape 列表
    shape_list = shapes_mini if run_mini else shapes_b1

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Layer', 'N', 'C', 'H', 'W', 'K', 'R', 'S', 'stride', 'padding', 'dilation',
            'Trials', 'Best Time (ms)', 'Std Dev (ms)', 'Tuning Time (min)', 'Status'
        ])

        for i, shape in enumerate(shape_list):
            N, C, H, W, K, _, R, S, _, stride, padding, dilation, groups = shape
            input_shape = (N, C, H, W)
            filter_shape = (K, C, R, S)
            strides = (stride, stride)
            paddings = (padding, padding)
            dilations = (dilation, dilation)

            # 每层独立日志目录
            layer_tag = f"layer_mini_{i}" if run_mini else f"layer_{i}"
            logfile = os.path.join(logfile_parent, layer_tag)

            # 清理旧目录
            if os.path.exists(logfile):
                shutil.rmtree(logfile)

            try:
                mod = create_conv2d_module(input_shape, filter_shape, strides, paddings, dilations, layout, dtype)
                mean_time, std_time, tuning_time = ms_execute(
                    mod, logfile, target, target_name, trials,
                    export_artifacts=True, stem=layer_tag
                )
                status = "Success"
            except Exception as e:
                print(f"An error occurred during benchmark for {layer_tag}: {e}")
                mean_time, std_time, tuning_time = -1, -1, -1
                status = "Failed"

            writer.writerow([
                layer_tag, N, C, H, W, K, R, S, stride, padding, dilation,
                trials, mean_time, std_time, tuning_time, status
            ])
            csvfile.flush()

    print(f"[done] CSV saved at: {csv_filename}")
