#!/usr/bin/env python3

"""
ResNet Metaschedule Benchmark Summary
This script analyzes and summarizes the available metaschedule benchmark results for ResNet models.
"""

import os
import json
import glob
from pathlib import Path

def analyze_benchmark_results():
    """Analyze existing benchmark results and create a summary."""
    
    base_dir = "/home/runner/work/bennu/bennu"
    results_dir = os.path.join(base_dir, "results")
    ms_results_dir = os.path.join(results_dir, "ms")
    
    print("=== ResNet Metaschedule Benchmark Analysis ===")
    print()
    
    # 1. Analyze ONNX models available
    models_dir = os.path.join(base_dir, "models")
    print("1. Available ONNX Models:")
    if os.path.exists(models_dir):
        for model in sorted(os.listdir(models_dir)):
            if model.endswith('.onnx'):
                model_path = os.path.join(models_dir, model)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"   - {model} ({size_mb:.1f} MB)")
    print()
    
    # 2. Analyze metaschedule results
    print("2. Available Metaschedule Results:")
    print()
    
    architectures = {}
    models = {}
    
    if os.path.exists(ms_results_dir):
        for result_dir in sorted(os.listdir(ms_results_dir)):
            if os.path.isdir(os.path.join(ms_results_dir, result_dir)):
                # Parse directory name: meta_{arch}_{device}_{model}_{trials}
                parts = result_dir.split('_')
                if len(parts) >= 4:
                    method = parts[0]  # meta
                    arch = parts[1]    # cuda, x86, arm
                    device = parts[2]  # a100, rtx3080, amd3700x, a64fx
                    model = parts[3]   # resnet18, alexnet, etc.
                    
                    # Group by architecture
                    arch_key = f"{arch}_{device}"
                    if arch_key not in architectures:
                        architectures[arch_key] = []
                    architectures[arch_key].append(model)
                    
                    # Group by model
                    if model not in models:
                        models[model] = []
                    models[model].append(arch_key)
    
    # Print by architecture
    print("Results by Architecture:")
    for arch, model_list in sorted(architectures.items()):
        print(f"   {arch.upper()}:")
        resnet_models = [m for m in sorted(set(model_list)) if 'resnet' in m]
        other_models = [m for m in sorted(set(model_list)) if 'resnet' not in m]
        
        if resnet_models:
            print(f"     ResNet: {', '.join(resnet_models)}")
        if other_models:
            print(f"     Others: {', '.join(other_models[:5])}")
            if len(other_models) > 5:
                print(f"             ... and {len(other_models)-5} more")
        print()
    
    # Print ResNet-specific analysis
    print("3. ResNet Model Analysis:")
    print()
    resnet_variants = [m for m in models.keys() if 'resnet' in m]
    
    if resnet_variants:
        print("ResNet Variants Available:")
        for variant in sorted(resnet_variants):
            archs = sorted(set(models[variant]))
            print(f"   - {variant.upper()}: {', '.join(archs)}")
        print()
    
    # 3. Analyze Ansor/Droplet results in main results directory
    print("4. Ansor/Droplet Results (results/):")
    print()
    
    ansor_results = []
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json') and ('resnet' in file.lower() or 'ResNet' in file):
                ansor_results.append(file)
    
    if ansor_results:
        print("ResNet Ansor/Droplet JSON Results:")
        for result in sorted(ansor_results):
            result_path = os.path.join(results_dir, result)
            size_mb = os.path.getsize(result_path) / (1024 * 1024)
            print(f"   - {result} ({size_mb:.1f} MB)")
    print()
    
    # 4. Benchmark scripts analysis
    print("5. Available Benchmark Scripts:")
    scripts_dir = os.path.join(base_dir, "scripts")
    benchmark_scripts = []
    
    if os.path.exists(scripts_dir):
        for script in os.listdir(scripts_dir):
            if script.endswith('.sh') and ('bench' in script or 'meta' in script):
                benchmark_scripts.append(script)
    
    if benchmark_scripts:
        for script in sorted(benchmark_scripts):
            print(f"   - scripts/{script}")
    print()
    
    # 5. Example commands
    print("6. Quick Start Commands:")
    print()
    print("Run ResNet18 metaschedule benchmark:")
    print("   python3 benchmarks/models_onnx.py -m meta -a x86 -t 1000 \\")
    print("           -l results/my_resnet18_test -b models/resnet18.onnx")
    print()
    print("Run complete benchmark suite:")
    print("   ./scripts/meta_benchmarks.sh")
    print()
    print("Analyze existing results:")
    print("   cat results/ms/meta_cuda_a100_resnet18_10k/output.txt")
    print()

def main():
    """Main function."""
    try:
        analyze_benchmark_results()
        print("For detailed documentation, see METASCHEDULE_BENCHMARKS.md")
        print("For interactive demo, run: ./run_resnet_benchmarks.sh")
    except Exception as e:
        print(f"Error analyzing benchmarks: {e}")

if __name__ == "__main__":
    main()