# Metaschedule Benchmarks in Bennu

This document provides a comprehensive guide to the metaschedule benchmarks available in the Bennu repository, with a focus on ResNet and other neural network models.

## Overview

Bennu combines [Ansor](https://tvm.apache.org/2021/03/03/intro-auto-scheduler) and [AutoTVM](https://tvm.apache.org/docs/reference/api/python/autotvm.html) optimization techniques. The repository includes extensive metaschedule benchmarking infrastructure using TVM's Meta Schedule framework.

## Key Components

### 1. Main Benchmark Scripts

#### `benchmarks/models_onnx.py`
Primary script for running ONNX model benchmarks with metaschedule support.

**Methods supported:**
- `ansor` - Pure Ansor autoscheduler
- `meta` - Pure metaschedule tuning
- `dpmeta` - Droplet + Metaschedule hybrid approach
- `droplet` - Droplet search optimization
- `run` - Execute optimized models

**Usage examples:**
```bash
# Generate metaschedule template with 10k trials for ResNet18 on x86
python3 benchmarks/models_onnx.py -m meta -a x86 -t 10000 -l results/meta_x86_resnet18_10k -b models/resnet18.onnx

# Generate metaschedule template for ResNet18 on CUDA
python3 benchmarks/models_onnx.py -m meta -a cuda -t 10000 -l results/meta_cuda_resnet18_10k -b models/resnet18.onnx

# Run DPMeta (Droplet + Metaschedule) optimization
python3 benchmarks/models_onnx.py -m dpmeta -a x86 -t 100 -k 1000 -l results/meta_x86_resnet18_10k -b models/resnet18.onnx
```

#### `benchmarks/models.py`
Script for running benchmarks using TVM relay testing models (programmatically generated models).

**Supported models:**
- ResNet (18, 34, 50, 101, 152 layers)
- MobileNet
- SqueezeNet v1.1
- Inception v3
- BERT
- VGG variants

**Usage examples:**
```bash
# Generate Ansor template for ResNet-50
python3 benchmarks/models.py -m ansor -a x86 -l results/resnet50.json -b resnet-50

# Run Droplet optimization on existing Ansor results
python3 benchmarks/models.py -m droplet -a x86 -l results/resnet50.json -b resnet-50 -t 100
```

### 2. Automated Benchmark Scripts

#### `scripts/meta_benchmarks.sh`
Automated script to run metaschedule benchmarks for all supported models.

**Supported models include:**
- alexnet, vgg11, vgg13, vgg16, vgg19
- **resnet18, resnet34, resnet50, resnet101, resnet152**
- shufflenet, squeezenet
- mobilenet_v2, mnasnet1_0
- inception_v3, googlenet
- densenet121, densenet161, densenet169, densenet201

**Configuration:**
```bash
NAME="test"          # Change this to your experiment name
ARCH="x86"          # Options: x86, cuda, arm
trials=10000        # Number of tuning trials
```

#### `scripts/dpmeta_benchmarks.sh`
Script for running DPMeta (Droplet + Metaschedule) benchmarks with various top-k configurations.

### 3. Available ResNet Models

#### ONNX Models (in `/models/` directory):
- `resnet18.onnx`
- Additional models: `shufflenet.onnx`, `squeezenet.onnx`

#### Programmatic Models (via TVM Relay):
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- Generated dynamically using `relay.testing.resnet.get_workload()`

## Supported Architectures

### x86 (CPU)
```bash
target_name = "llvm"
target = tvm.target.Target("llvm")
```

### CUDA (GPU)
```bash
target_name = "cuda"
target = tvm.target.Target("cuda")
# OR with specific parameters
target = "cuda -max_threads_per_block 1024 -max_shared_memory_per_block 49152"
```

### ARM (A64FX)
```bash
target_name = "llvm -mcpu=a64fx"
target = tvm.target.Target("llvm -mcpu=a64fx")
```

## Existing Benchmark Results

The repository contains extensive pre-computed benchmark results in `/results/` and `/results/ms/` directories:

### ResNet Results Available:
- **CUDA A100:** ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **CUDA RTX3080:** ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **x86 AMD3700X:** Various ResNet models
- **ARM A64FX:** ResNet34 and others

### Result Structure:
Each metaschedule result directory contains:
- `database_tuning_record.json` - Tuning records
- `database_workload.json` - Workload definitions
- `logs/` - Detailed per-task logs
- `output.txt` - Summary and profiling information

### Example Result Locations:
```
/results/ms/meta_cuda_a100_resnet18_10k/
/results/ms/meta_cuda_a100_resnet50_10k/
/results/ms/meta_cuda_rtx3080_resnet18_10k/
/results/ms/meta_x86_amd3700x_alexnet_10k/
```

## Metaschedule Configuration

### Key Parameters:
- `max_trials_global` - Total number of tuning trials (default: 10,000)
- `num_trials_per_iter` - Trials per iteration (default: 64)
- `number` - Number of measurements per trial (default: 3)
- `repeat` - Number of repeats per measurement (default: 3)
- `min_repeat_ms` - Minimum repeat time in ms (default: 100)

### Cost Model:
```python
cost_model=ms.cost_model.XGBModel(
    extractor=ms.feature_extractor.PerStoreFeature(),
    adaptive_training=False,
)
```

### Search Strategy:
```python
strategy=ms.search_strategy.EvolutionarySearch()
```

## Running Your Own Benchmarks

### 1. Single ResNet Model (ONNX):
```bash
# ResNet18 on x86 with 1000 trials
python3 benchmarks/models_onnx.py -m meta -a x86 -t 1000 -l results/my_resnet18_test -b models/resnet18.onnx

# ResNet18 on CUDA with 10k trials
python3 benchmarks/models_onnx.py -m meta -a cuda -t 10000 -l results/my_resnet18_cuda -b models/resnet18.onnx
```

### 2. Multiple Models Using Scripts:
```bash
# Edit scripts/meta_benchmarks.sh to configure your experiment
# Set NAME, ARCH, and trials variables
./scripts/meta_benchmarks.sh
```

### 3. Hybrid DPMeta Approach:
```bash
# First generate metaschedule baseline with 10k trials
python3 benchmarks/models_onnx.py -m meta -a x86 -t 10000 -l results/meta_baseline -b models/resnet18.onnx

# Then run DPMeta optimization with 100 droplet trials on top-1000 results
python3 benchmarks/models_onnx.py -m dpmeta -a x86 -t 100 -k 1000 -l results/meta_baseline -b models/resnet18.onnx
```

## Performance Analysis

### Output Metrics:
- **Latency (μs):** Execution time per inference
- **Speed (GFLOPS):** Computational throughput
- **Tuning Time:** Time spent optimizing
- **Speedup:** Improvement over baseline

### Example Output:
```
Layer, DPMeta time(s), DPMeta std(s), DPMeta trials, DPMeta Tuning(min), Meta-1000 time(s), Meta-1000 std(s), speedup-1000
0, 0.0001234567, 0.0000123456, 100, 2.34, 0.0002345678, 0.0000234567, 1.90
```

## Integration with Other Tools

### Droplet Search (`src/DropletSearch.py`):
Advanced search algorithm that can be combined with metaschedule results.

### Grid Search (`src/GridSearch.py`):
Systematic parameter space exploration.

### XGBoost Search (`src/XGBSearch.py`):
Machine learning-guided optimization.

## Quick Start for ResNet Benchmarking

1. **Install dependencies** (TVM >= 0.13, Python >= 3.6)

2. **Run a simple ResNet18 metaschedule benchmark:**
```bash
python3 benchmarks/models_onnx.py -m meta -a x86 -t 100 -l test_resnet18 -b models/resnet18.onnx
```

3. **Check the results:**
```bash
ls test_resnet18/
cat test_resnet18/output.txt
```

4. **Run the full benchmark suite:**
```bash
# Edit scripts/meta_benchmarks.sh first
./scripts/meta_benchmarks.sh
```

## Notes

- Metaschedule tuning can be time-intensive (hours for 10k trials)
- CUDA benchmarks require appropriate GPU hardware
- Results are highly dependent on the target hardware configuration
- The repository includes extensive pre-computed results for comparison