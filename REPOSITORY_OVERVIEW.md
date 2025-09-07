# Bennu Repository Documentation

## Project Overview
Bennu combines the TVM Ansor auto-scheduler with AutoTVM so that Ansor explores kernel configurations and AutoTVM exploits the best solutions discovered in the search. It requires TVM ≥ 0.13 and Python ≥ 3.6.

## Directory Tree
```
bennu/
|-- assets/
|   `-- img/                       - project banners
|-- benchmarks/
|   |-- conv2d.py
|   |-- conv3d.py
|   |-- depthwise.py
|   |-- mm.py
|   |-- models.py
|   |-- models_onnx.py
|   |-- models_onnx_cache.py
|   |-- models_onnx_compare.py
|   |-- pooling.py
|   |-- relu.py
|   `-- microkernels/
|       |-- DpAnsor/
|       |   |-- a100/, log/, results/
|       |   |-- run scripts
|       |   `-- src/                - dpansor algorithm utilities
|       |-- ansor/                  - Ansor microkernel tuners (src/...)
|       |-- autotvm/                - AutoTVM microkernel tuners (src/...)
|       |-- droplet/                - Droplet search microkernel tuners (src/...)
|       |-- heron/                  - Heron reference kernels (run.py, src/)
|       |-- ms/                     - Meta-schedule microkernel examples (src/)
|       |-- original/               - Original Ansor templates (src/, utils/)
|       |-- pytorch/                - PyTorch baseline kernels (src/)
|       |-- roller/                 - Roller framework tests (src/)
|       `-- tf/                     - TensorFlow baseline kernels (src/)
|-- docker/
|   |-- cuda/                       - Dockerfiles for CUDA environment
|   `-- x86/                        - Dockerfiles for x86 environment
|-- models/                         - ONNX models (resnet18, shufflenet, squeezenet)
|-- perf_stats/                     - Collected performance statistics
|-- results/                        - Generated tuning logs (JSON/CSV) and meta-schedule outputs
|-- scripts/                        - Shell helpers for batch experiments
|-- src/
|   |-- DPMeta.py
|   |-- DropletSearch.py
|   |-- GASearch.py
|   |-- GridSearch.py
|   |-- RandomSearch.py
|   |-- XGBSearch.py
|   |-- measure_time.py
|   |-- plot_figure.py
|   |-- space.py
|   |-- space_ms.py
|   `-- utils.py
`-- utils/
    |-- README.md
    `-- print_record_info.py
```

## Python Scripts

| Path(s) | Purpose & Functionality | Inputs / Outputs | Category |
| --- | --- | --- | --- |
| `benchmarks/conv2d.py` | Defines and tunes a 2-D convolution workload with TVM’s auto-scheduler; can fall back to Droplet tuning | CLI args: method, arch, logfile, trials; outputs tuning log and timing prints | Experiment runner |
| `benchmarks/conv3d.py` | 3-D convolution benchmark with Ansor or Droplet optimization | Same CLI options; emits search log | Experiment runner |
| `benchmarks/depthwise.py` | Depthwise convolution benchmarking and log generation | CLI params; writes tuning log | Experiment runner |
| `benchmarks/mm.py` | Matrix-multiplication tuning harness | CLI; writes log of best configs | Experiment runner |
| `benchmarks/models.py` | Extracts Relay networks (ResNet, BERT, etc.) and tunes them; supports Droplet exploitation | Model name, architecture, log file; produces per-layer logs and timing | Experiment runner |
| `benchmarks/models_onnx.py` | Loads ONNX models and tunes via Ansor, Meta-Schedule, or Droplet; also supports meta-schedule tuning and Droplet exploitation | ONNX model path, target, trials; outputs tuning database and stats | Experiment runner |
| `benchmarks/models_onnx_cache.py` | Same as above but reuses cached best configs for faster experiments | ONNX model, log file, top-K; outputs droplet logs and comparisons | Experiment runner |
| `benchmarks/models_onnx_compare.py` | Compares different search algorithms (Droplet, Grid, GA, XGB, Random) on ONNX models | ONNX model, algorithm choices; prints timing comparisons | Experiment runner |
| `benchmarks/pooling.py` | Benchmarks average pooling kernels | CLI; log file | Experiment runner |
| `benchmarks/relu.py` | ReLU elementwise benchmark | CLI; log file | Experiment runner |
| `benchmarks/microkernels/DpAnsor/src/dpansor.py` | Runs DPAnsor (Droplet + Ansor) for microkernels (matmul, conv2d, depthwise, etc.) | Arch, log file, benchmark, method; emits comparison stats | Experiment runner |
| `benchmarks/microkernels/DpAnsor/src/utils.py`, `space.py`, `DropletSearch.py`, `print_output.py` | Helpers for DPAnsor: search algorithm, config space and result formatting | internal JSON configs and logs; print summaries | Utilities |
| `benchmarks/microkernels/ansor/src/conv2d_tuning.py`, `depthwise_tuning.py`, `matmul_tuning.py`, `pooling_tuning.py`, `reduction_tuning.py`, `relu_tuning.py` | Auto-scheduler tuning scripts for individual microkernels using TVM Ansor | Workload parameters, log paths; output Ansor logs | Experiment runners |
| `benchmarks/microkernels/ansor/src/utils/*` | CUDA kernels, parsers, best-config utilities for Ansor microkernels | Takes log files/configs; returns best candidates or executes kernels | Utilities |
| `benchmarks/microkernels/autotvm/src/*.py` | Equivalent microkernel tuners using AutoTVM (conv2d, depthwise, etc.) | Operation parameters; produces AutoTVM logs | Experiment runners |
| `benchmarks/microkernels/droplet/src/*.py` | Droplet search versions of microkernel tuners | Operation parameters; Droplet tuning logs | Experiment runners |
| `benchmarks/microkernels/heron/src/conv2d.py`, `gemm.py` | Uses PyTorch to measure baseline conv2d/GEMM performance for comparison | Hard-coded tensor sizes; prints timing | Analysis tools |
| `benchmarks/microkernels/ms/src/*.py` | Microkernel examples built with TVM Meta-Schedule | Operation parameters; tuning logs | Experiment runners |
| `benchmarks/microkernels/original/src/*.py` | Original Ansor templates prior to Bennu’s modifications | Operation parameters; logs | Experiment runners |
| `benchmarks/microkernels/original/utils/*.py` | Utility CUDA kernels and parsers for original templates | Log files, configs; prints | Utilities |
| `benchmarks/microkernels/pytorch/src/*.py`, `tf/src/*.py` | Baseline implementations in PyTorch and TensorFlow for runtime comparison | Fixed shapes; print timing | Analysis tools |
| `benchmarks/microkernels/roller/src/test_*.py` | Benchmark scripts for the Roller framework measuring various ops | none; prints compile/runtime stats | Analysis tools |
| `src/DPMeta.py` | Implements DropletMeta search over Meta-Schedule spaces | JSON workload/config, target; writes tuning log | Search algorithm |
| `src/DropletSearch.py` | Core Droplet search algorithm operating on Ansor spaces | JSON template, target, log; writes iterative results | Search algorithm |
| `src/GASearch.py` | Genetic algorithm tuner for Ansor spaces | JSON template, population parameters; tuning log | Search algorithm |
| `src/GridSearch.py` | Exhaustive grid search across Ansor space | JSON template, log; writes evaluated configs | Search algorithm |
| `src/RandomSearch.py` | Random sampling of Ansor space | JSON template, log; random trials saved | Search algorithm |
| `src/XGBSearch.py` | Model-based tuner using XGBoost cost model | JSON template, optimizer params; produces logs | Search algorithm |
| `src/space.py` | Defines tunable configuration space and runner for Ansor logs | JSON config & SearchTask; runs variants, saves records | Utility |
| `src/space_ms.py` | Analogous space definition for TVM Meta-Schedule | JSON workload & target; generates config search space | Utility |
| `src/utils.py` | File/log helpers, timing extraction, permutations, etc. | Reads/writes JSON logs; returns best times/configs | Utility |
| `src/measure_time.py` | Reads meta-schedule logs and prints per-layer timings | Path to tuning file; outputs layer statistics | Analysis tool |
| `src/plot_figure.py` | Creates speedup graphs from CSV data | CSV of results; plots matplotlib figure | Analysis tool |
| `utils/print_record_info.py` | Displays detailed info for a specific tuning record (e.g., DAG, FLOPs) | logfile path and index; prints record summary | Utility |

## Data/Config/Results Folders

| Folder | Contents & Role | Type |
| --- | --- | --- |
| `models/` | Pretrained ONNX models used as raw inputs for tuning (e.g., resnet18.onnx, shufflenet.onnx) | Raw inputs |
| `benchmarks/microkernels/*/results/` | CSV/JSON logs produced by microkernel tuning runs for various backends | Generated logs |
| `results/` | Central store for tuning logs from benchmarks and meta-schedule runs; includes JSON task records and CSV summaries | Generated logs |
| `perf_stats/` | Collected performance statistics for different machines (e.g., RTX3080, a64fx) | Measurement outputs |
| `assets/img/` | Banner images for documentation | Static assets |
| `scripts/` | Shell scripts orchestrating experiments or building environments; they are templates, not data | Utilities |
| `docker/` | Dockerfiles and helper scripts to set up x86/CUDA environments | Environment configs |

## Suggested Usage Flow
1. **Select Benchmark**: Choose a kernel or model (e.g., matrix multiply, ResNet18).
2. **Generate Ansor Template**: Run one of the `benchmarks/*.py` scripts with `-m ansor` to produce an initial log file. Example:
   ```bash
   python3 benchmarks/mm.py -m ansor -a x86 -l results/x86_mm.json
   ```
3. **Optimize with Search Algorithm**: Use the same script with another method (`droplet`, `dpansor`, or meta-schedule options) to exploit the template and refine configurations.
4. **Inspect Results**: Logs appear under `results/` or `benchmarks/microkernels/*/results/`. Tools like `utils/print_record_info.py` show details for individual records, while `src/plot_figure.py` or spreadsheets analyze speedup graphs.
5. **Compare Baselines**: Optional scripts in microkernel subfolders (e.g., PyTorch or TensorFlow baselines) allow runtime comparisons against tuned kernels.
