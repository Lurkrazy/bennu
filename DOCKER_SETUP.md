# Docker Setup for TVM 0.13.0 + PyTorch + CUDA 12.4

This directory contains a complete Docker workflow for running TVM 0.13.0 metaschedule benchmarks with PyTorch and CUDA 12.4 support.

## 🚀 Quick Start

```bash
# 1. Build the Docker image
./docker-workflow.sh build

# 2. Start container in background
./docker-workflow.sh run

# 3. Test TVM installation
./docker-workflow.sh test-tvm

# 4. Test PyTorch installation  
./docker-workflow.sh test-pytorch

# 5. Attach to container for interactive work
./docker-workflow.sh attach
```

## 📋 Prerequisites

- Docker with GPU support (nvidia-docker2)
- NVIDIA drivers compatible with CUDA 12.4
- At least 8GB free disk space

## 🛠️ Available Commands

### Image Management
```bash
./docker-workflow.sh build              # Build Docker image
./docker-workflow.sh status             # Check container status
./docker-workflow.sh cleanup            # Remove container
```

### Container Operations
```bash
./docker-workflow.sh run               # Start container in background
./docker-workflow.sh attach            # Attach to running container
./docker-workflow.sh interactive       # Run one-time interactive session
```

### Testing & Verification
```bash
./docker-workflow.sh test-tvm          # Verify TVM installation
./docker-workflow.sh test-pytorch      # Verify PyTorch + CUDA
./docker-workflow.sh benchmark         # Run demo ResNet benchmark
```

## 🔧 Manual Docker Commands

If you prefer manual control:

### Build Image
```bash
docker build -t bennu-tvm:cuda12.4 .
```

### Run Container (Background)
```bash
docker run --gpus all -d -it \
    --name bennu-benchmark \
    -v $(pwd):/root/bennu \
    bennu-tvm:cuda12.4
```

### Attach to Container
```bash
docker exec -it bennu-benchmark /bin/bash
```

### Run One-Time Interactive
```bash
docker run --gpus all -it --rm \
    -v $(pwd):/root/bennu \
    bennu-tvm:cuda12.4
```

## 🧪 Testing TVM Installation

Inside the container:

```bash
# Test basic TVM functionality
python3 -c "import tvm; print('TVM version:', tvm.__version__)"

# Test Meta Schedule API
python3 -c "import tvm.meta_schedule; print('Meta Schedule available')"

# Test CUDA support
python3 -c "import tvm; print('CUDA available:', tvm.cuda().exist)"
```

## 🔥 Testing PyTorch + CUDA

Inside the container:

```bash
# Test PyTorch installation
python3 -c "import torch; print('PyTorch version:', torch.__version__)"

# Test CUDA availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Test GPU detection
python3 -c "import torch; print('GPU count:', torch.cuda.device_count())"
```

## 📊 Running Benchmarks

### Quick ResNet-18 Benchmark
```bash
# Inside container
cd /root/bennu
python3 benchmarks/models_onnx.py -m meta -a cuda -t 1000 \
    -l results/resnet18_test -b models/resnet18.onnx
```

### DPMeta Hybrid Optimization
```bash
# Step 1: Generate baseline
python3 benchmarks/models_onnx.py -m meta -a cuda -t 10000 \
    -l results/resnet18_baseline -b models/resnet18.onnx

# Step 2: Run hybrid optimization
python3 benchmarks/models_onnx.py -m dpmeta -a cuda -t 100 -k 1000 \
    -l results/resnet18_baseline -b models/resnet18.onnx
```

### Analysis Tools
```bash
# Analyze all benchmark results
python3 analyze_resnet_benchmarks.py

# Interactive benchmark exploration
./run_resnet_benchmarks.sh
```

## 🏗️ Dockerfile Features

- **Base**: NVIDIA CUDA 12.4.1 on Ubuntu 22.04
- **TVM**: v0.13.0 with CUDA support and Meta Schedule API
- **PyTorch**: Latest version with CUDA 12.4 support
- **Dependencies**: All required packages for metaschedule benchmarking
- **Optimizations**: Multi-core compilation, proper environment setup

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Container Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check for port conflicts
./docker-workflow.sh cleanup
./docker-workflow.sh run
```

### Build Failures
```bash
# Clean build
docker system prune -f
./docker-workflow.sh build
```

## 📁 Volume Mounts

The container automatically mounts the current directory to `/root/bennu`, providing:
- Access to benchmark scripts
- Persistent results storage
- Easy file sharing between host and container

## 🎯 Performance Tips

1. **Multi-core builds**: The Dockerfile uses `make -j$(nproc)` for faster compilation
2. **GPU memory**: Ensure sufficient GPU memory for large models
3. **Persistent container**: Use background mode for long-running benchmarks
4. **Results storage**: Keep results in the mounted volume for persistence

## 📚 Additional Resources

- [TVM Documentation](https://tvm.apache.org/docs/)
- [Meta Schedule Tutorial](https://tvm.apache.org/docs/tutorial/meta_schedule_index.html)
- [Bennu Benchmark Documentation](METASCHEDULE_BENCHMARKS.md)
- [DPMeta Hybrid Optimization Guide](run_resnet_benchmarks.sh)