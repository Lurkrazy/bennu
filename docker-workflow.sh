#!/bin/bash

# Docker Workflow for TVM 0.13.0 + PyTorch + CUDA 12.4 Benchmarking
# This script provides a complete workflow for setting up and using the benchmarking environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
echo_success() { echo -e "${GREEN}✅ $1${NC}"; }
echo_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
echo_error() { echo -e "${RED}❌ $1${NC}"; }

# Configuration
IMAGE_NAME="bennu-tvm"
TAG="cuda12.4"
CONTAINER_NAME="bennu-benchmark"

# Function to build the Docker image
build_image() {
    echo_info "Building Docker image: ${IMAGE_NAME}:${TAG}"
    docker build -t "${IMAGE_NAME}:${TAG}" .
    echo_success "Docker image built successfully!"
}

# Function to run container in background
run_container_background() {
    echo_info "Starting container in background: ${CONTAINER_NAME}"
    
    # Stop existing container if running
    if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo_warning "Stopping existing container: ${CONTAINER_NAME}"
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    fi
    
    # Start new container
    docker run --gpus all -d -it \
        --name "${CONTAINER_NAME}" \
        -v "$(pwd):/root/bennu" \
        "${IMAGE_NAME}:${TAG}"
    
    echo_success "Container started in background!"
    echo_info "Attach with: docker exec -it ${CONTAINER_NAME} /bin/bash"
    echo_info "Stop with: docker stop ${CONTAINER_NAME}"
}

# Function to attach to running container
attach_container() {
    if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo_info "Attaching to running container: ${CONTAINER_NAME}"
        docker exec -it "${CONTAINER_NAME}" /bin/bash
    else
        echo_error "Container ${CONTAINER_NAME} is not running!"
        echo_info "Start it first with: $0 run"
        exit 1
    fi
}

# Function to run container interactively (one-time)
run_container_interactive() {
    echo_info "Running container interactively (one-time)"
    docker run --gpus all -it --rm \
        -v "$(pwd):/root/bennu" \
        "${IMAGE_NAME}:${TAG}"
}

# Function to test TVM installation
test_tvm() {
    echo_info "Testing TVM installation..."
    docker exec -it "${CONTAINER_NAME}" python3 -c "
import tvm
import tvm.meta_schedule
print('✅ TVM version:', tvm.__version__)
print('✅ TVM Meta Schedule available')
print('✅ TVM CUDA support:', tvm.cuda().exist)
"
    echo_success "TVM test completed!"
}

# Function to test PyTorch installation
test_pytorch() {
    echo_info "Testing PyTorch installation..."
    docker exec -it "${CONTAINER_NAME}" python3 -c "
import torch
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU count:', torch.cuda.device_count())
    print('✅ Current GPU:', torch.cuda.get_device_name(0))
"
    echo_success "PyTorch test completed!"
}

# Function to run ResNet benchmark demo
run_benchmark_demo() {
    echo_info "Running ResNet benchmark demo..."
    docker exec -it "${CONTAINER_NAME}" bash -c "
cd /root/bennu
echo '🔍 Available models:'
ls -la models/ | grep resnet || echo 'No ResNet models found - download them first'
echo ''
echo '📊 Running quick ResNet-18 benchmark with Meta Schedule:'
python3 benchmarks/models_onnx.py -m meta -a cuda -t 100 -l results/demo_resnet18 -b models/resnet18.onnx || echo 'Download ResNet models first'
echo ''
echo '📈 Available analysis tools:'
echo '  - python3 analyze_resnet_benchmarks.py'
echo '  - ./run_resnet_benchmarks.sh'
"
    echo_success "Benchmark demo completed!"
}

# Function to show container status
status() {
    echo_info "Container status:"
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "^${CONTAINER_NAME}"; then
        echo_success "Container ${CONTAINER_NAME} is running"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | grep "${CONTAINER_NAME}"
    else
        echo_warning "Container ${CONTAINER_NAME} is not running"
        if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo_info "Container exists but is stopped"
            docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | grep "${CONTAINER_NAME}"
        else
            echo_info "Container does not exist"
        fi
    fi
}

# Function to clean up
cleanup() {
    echo_info "Cleaning up Docker resources..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    echo_success "Cleanup completed!"
}

# Function to show usage
usage() {
    echo "Docker Workflow for TVM 0.13.0 + PyTorch + CUDA 12.4"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build              Build the Docker image"
    echo "  run                Start container in background"
    echo "  attach             Attach to running container"
    echo "  interactive        Run container interactively (one-time)"
    echo "  test-tvm           Test TVM installation"
    echo "  test-pytorch       Test PyTorch installation"
    echo "  benchmark          Run ResNet benchmark demo"
    echo "  status             Show container status"
    echo "  cleanup            Stop and remove container"
    echo "  help               Show this help message"
    echo ""
    echo "Complete Workflow:"
    echo "  1. $0 build                 # Build image"
    echo "  2. $0 run                   # Start container"
    echo "  3. $0 test-tvm              # Verify TVM"
    echo "  4. $0 test-pytorch          # Verify PyTorch"
    echo "  5. $0 attach                # Work in container"
    echo "  6. $0 benchmark             # Run benchmarks"
    echo ""
    echo "Quick Start:"
    echo "  $0 build && $0 run && $0 test-tvm && $0 test-pytorch"
}

# Main command dispatch
case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        run_container_background
        ;;
    attach)
        attach_container
        ;;
    interactive)
        run_container_interactive
        ;;
    test-tvm)
        test_tvm
        ;;
    test-pytorch)
        test_pytorch
        ;;
    benchmark)
        run_benchmark_demo
        ;;
    status)
        status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo_error "Unknown command: $1"
        echo ""
        usage
        exit 1
        ;;
esac