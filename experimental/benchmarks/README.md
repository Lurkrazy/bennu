# Project Overview

This project contains scripts and data for analyzing and tuning deep learning operators (mainly conv2d) on an NVIDIA 4090 GPU. It includes performance analysis, tuning logs, results, and visualization tools.

---

## Top-Level Scripts

- [`analyze_all_layers_gflops.py`](analyze_all_layers_gflops.py:1): Analyzes GFLOPS performance across all layers and summarizes tuning results.
- [`conv2d.py`](conv2d.py:1): Runs tuning or performance tests for the conv2d operator.
- [`conv2d.log`](conv2d.log:1): Log output from conv2d tuning or testing.
- [`export_times_to_csv.py`](export_times_to_csv.py:1): Exports timing data from experiments/tuning to CSV for further analysis.
- [`get_best_config.py`](get_best_config.py:1): Selects and outputs the best tuning configuration.
- [`get_best_tile_sizes.py`](get_best_tile_sizes.py:1): Analyzes and outputs the best tile size parameters for operator tuning.
- [`get_best_tile_sizes.log`](get_best_tile_sizes.log:1): Log output from tile size analysis.
- [`parse_trials.py`](parse_trials.py:1): Parses raw tuning experiment (trial) data.
- [`parse_trials_logs.py`](parse_trials_logs.py:1): Parses tuning logs and extracts key information.
- [`plot_best_gflops.py`](plot_best_gflops.py:1): Visualizes the best GFLOPS results for each layer.

---

## Folder Structure

- [`convergence/`](convergence/): Contains GFLOPS curve images showing tuning convergence for each layer (layer_0~layer_9).
- [`distribution/`](distribution/):
  - [`analyze_gflops.py`](distribution/analyze_gflops.py:1): Analyzes GFLOPS distribution.
  - [`combined_gflops_distribution.png`](distribution/combined_gflops_distribution.png:1), [`combined_gflops_kde.png`](distribution/combined_gflops_kde.png:1): Visualizations of GFLOPS distribution.
  - [`gflops_combined.csv`](distribution/gflops_combined.csv:1), [`gflops_stats_summary.csv`](distribution/gflops_stats_summary.csv:1): GFLOPS statistics data.
  - [`layer_X_gflops_hist.png`](distribution/layer_0_gflops_hist.png:1), etc.: GFLOPS histograms for each layer.

- [`layer_0/` ~ `layer_9/`]: Tuning data and logs for each layer
  - `database_tuning_record.json`: Tuning records
Each layer directory contains a `database_tuning_record.json` file, which records the kernel tuning experiment data for that layer. (Normally there are 1000 lines(trials) per layer)
- Each line(trial) represents the tuning result for a kernel, including:
  - The schedule information for the kernel
  - The corresponding tile size parameters
  - The execution time of the kernel
  - Relevant hardware information (such as GPU model and configuration)

These records are essential for analyzing and comparing the impact of different scheduling strategies and parameter configurations on performance.
  - `database_workload.json`: Workload information
  - `parsed_trials.csv`: Parsed experiment data
  - `times.csv`: Timing for each experiment
  - `logs/`: TVM scheduler logs

- [`results/`](results/): Final results summary (e.g., conv2d_results_*.csv)

---

## Suggested Workflow

1. Run tuning scripts (e.g., conv2d.py) to generate raw data and logs.
2. Use parsing scripts (e.g., parse_trials.py, parse_trials_logs.py) to process raw data.
3. Use analysis scripts (e.g., analyze_all_layers_gflops.py, analyze_gflops.py) to summarize performance.
4. Use visualization scripts (e.g., plot_best_gflops.py) to generate charts for result interpretation.
