#!/bin/bash

# ResNet Metaschedule Benchmark Quick Start
# This script demonstrates how to run metaschedule benchmarks for ResNet models

echo "=== ResNet Metaschedule Benchmark Quick Start ==="
echo ""

# Configuration
ARCH="x86"  # Change to "cuda" if you have GPU, or "arm" for ARM processors
TRIALS=100  # Increase to 1000 or 10000 for better results (will take longer)
BASE_DIR="results/quick_resnet_test"

echo "Configuration:"
echo "  Architecture: $ARCH"
echo "  Trials: $TRIALS"
echo "  Output directory: $BASE_DIR"
echo ""

# Create results directory
mkdir -p "$BASE_DIR"

echo "=== Available ResNet Benchmarks ==="
echo ""

echo "1. ResNet18 Metaschedule Benchmark (ONNX model):"
echo "   python3 benchmarks/models_onnx.py -m meta -a $ARCH -t $TRIALS -l $BASE_DIR/resnet18_meta -b models/resnet18.onnx"
echo ""

echo "2. ResNet50 Relay Model Benchmark:"
echo "   python3 benchmarks/models.py -m ansor -a $ARCH -t $TRIALS -l $BASE_DIR/resnet50_ansor.json -b resnet-50"
echo ""

echo "3. Batch ResNet Benchmarks (all variants):"
echo "   # Edit scripts/meta_benchmarks.sh to set ARCH=$ARCH and trials=$TRIALS"
echo "   ./scripts/meta_benchmarks.sh"
echo ""

echo "=== Running Quick ResNet18 Demo ==="
echo ""

# Check if models exist
if [ ! -f "models/resnet18.onnx" ]; then
    echo "Warning: models/resnet18.onnx not found. Please ensure ONNX models are available."
    echo "Available models in models/ directory:"
    ls -la models/ 2>/dev/null || echo "models/ directory not found"
    echo ""
fi

# Run a quick demo (commented out to avoid long execution during exploration)
echo "To run a quick ResNet18 metaschedule demo:"
echo "python3 benchmarks/models_onnx.py -m meta -a $ARCH -t $TRIALS -l $BASE_DIR/resnet18_demo -b models/resnet18.onnx"
echo ""

echo "=== Analyzing Existing Results ==="
echo ""

echo "Pre-computed ResNet results are available in:"
echo ""

# List existing ResNet results
echo "Metaschedule results (results/ms/):"
find results/ms/ -name "*resnet*" -type d 2>/dev/null | head -10 | while read dir; do
    echo "  - $dir"
done

echo ""
echo "Ansor/Droplet results (results/):"
find results/ -name "*resnet*" -maxdepth 1 2>/dev/null | head -5 | while read file; do
    echo "  - $file"
done

echo ""
echo "=== Example Commands by Architecture ==="
echo ""

echo "For x86 (CPU):"
echo "  python3 benchmarks/models_onnx.py -m meta -a x86 -t 1000 -l results/resnet18_x86 -b models/resnet18.onnx"
echo ""

echo "For CUDA (GPU):"
echo "  python3 benchmarks/models_onnx.py -m meta -a cuda -t 1000 -l results/resnet18_cuda -b models/resnet18.onnx"
echo ""

echo "For ARM (A64FX):"
echo "  python3 benchmarks/models_onnx.py -m meta -a arm -t 1000 -l results/resnet18_arm -b models/resnet18.onnx"
echo ""

echo "=== DPMeta (Droplet + Metaschedule) Workflow ==="
echo ""

echo "1. First generate metaschedule baseline:"
echo "   python3 benchmarks/models_onnx.py -m meta -a $ARCH -t 10000 -l $BASE_DIR/meta_baseline -b models/resnet18.onnx"
echo ""

echo "2. Then run DPMeta optimization:"
echo "   python3 benchmarks/models_onnx.py -m dpmeta -a $ARCH -t 100 -k 1000 -l $BASE_DIR/meta_baseline -b models/resnet18.onnx"
echo ""

echo "=== Result Analysis ==="
echo ""

echo "After running benchmarks, check results with:"
echo "  ls -la \$RESULT_DIR/"
echo "  cat \$RESULT_DIR/output.txt"
echo "  # For metaschedule results:"
echo "  ls -la \$RESULT_DIR/logs/"
echo ""

echo "=== Performance Comparison ==="
echo ""

echo "To compare different methods on the same model:"
echo "1. Ansor:   python3 benchmarks/models_onnx.py -m ansor -a $ARCH -t $TRIALS -l $BASE_DIR/ansor -b models/resnet18.onnx"
echo "2. Meta:    python3 benchmarks/models_onnx.py -m meta -a $ARCH -t $TRIALS -l $BASE_DIR/meta -b models/resnet18.onnx"
echo "3. Droplet: python3 benchmarks/models_onnx.py -m droplet -a $ARCH -t 100 -l $BASE_DIR/ansor -b models/resnet18.onnx"
echo ""

echo "=== Complete Benchmark Suite ==="
echo ""

echo "To run the complete benchmark suite for all ResNet variants:"
echo "1. Edit scripts/meta_benchmarks.sh:"
echo "   - Set NAME to your experiment name"
echo "   - Set ARCH to your target architecture"
echo "   - Set trials to desired number (1000-10000)"
echo ""
echo "2. Run: ./scripts/meta_benchmarks.sh"
echo ""

echo "This will benchmark: alexnet, vgg11, resnet18, resnet34, resnet50, resnet101, resnet152, and more"
echo ""

echo "For more details, see METASCHEDULE_BENCHMARKS.md"