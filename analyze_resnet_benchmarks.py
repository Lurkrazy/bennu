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
    
    # 3. Analyze DPMeta (Droplet + Metaschedule) hybrid results
    print("4. DPMeta (Droplet Hybrid) Analysis:")
    print()
    
    dpmeta_logs = []
    if os.path.exists(base_dir):
        # Look for layer_*.log files which indicate DPMeta runs
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.startswith('layer_') and file.endswith('.log'):
                    dpmeta_logs.append(os.path.join(root, file))
    
    if dpmeta_logs:
        print("DPMeta Hybrid Optimization Results Found:")
        log_dirs = set(os.path.dirname(log) for log in dpmeta_logs)
        for log_dir in sorted(log_dirs):
            rel_path = os.path.relpath(log_dir, base_dir)
            layer_count = len([f for f in os.listdir(log_dir) if f.startswith('layer_') and f.endswith('.log')])
            print(f"   - {rel_path}/ ({layer_count} layers analyzed)")
        print()
        print("DPMeta combines TVM's Meta Schedule with Droplet search for hybrid optimization.")
        print("Each layer_*.log file contains Droplet search results for individual neural network layers.")
    else:
        print("No DPMeta hybrid optimization results found.")
        print("To run DPMeta hybrid optimization:")
        print("  1. First generate metaschedule baseline: python3 benchmarks/models_onnx.py -m meta ...")
        print("  2. Then run DPMeta: python3 benchmarks/models_onnx.py -m dpmeta ...")
    print()
    
    # 4. Analyze Ansor/Droplet results in main results directory
    print("5. Ansor/Droplet Results (results/):")
    print()
    
    ansor_results = []
    droplet_results = []
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json') and ('resnet' in file.lower() or 'ResNet' in file):
                ansor_results.append(file)
            elif file.endswith('_droplet.json'):
                droplet_results.append(file)
    
    if ansor_results:
        print("ResNet Ansor/Traditional Results:")
        for result in sorted(ansor_results):
            result_path = os.path.join(results_dir, result)
            size_mb = os.path.getsize(result_path) / (1024 * 1024)
            print(f"   - {result} ({size_mb:.1f} MB)")
    
    if droplet_results:
        print()
        print("Droplet Search Results:")
        for result in sorted(droplet_results):
            result_path = os.path.join(results_dir, result)
            size_mb = os.path.getsize(result_path) / (1024 * 1024)
            print(f"   - {result} ({size_mb:.1f} MB)")
    print()
    
    # 6. Benchmark scripts analysis
    print("6. Available Benchmark Scripts:")
    scripts_dir = os.path.join(base_dir, "scripts")
    benchmark_scripts = []
    
    if os.path.exists(scripts_dir):
        for script in os.listdir(scripts_dir):
            if script.endswith('.sh') and ('bench' in script or 'meta' in script or 'droplet' in script):
                benchmark_scripts.append(script)
    
    if benchmark_scripts:
        print("Metaschedule & Hybrid Optimization Scripts:")
        for script in sorted(benchmark_scripts):
            if 'dpmeta' in script:
                print(f"   - scripts/{script} (DPMeta hybrid optimization)")
            elif 'droplet' in script:
                print(f"   - scripts/{script} (Droplet search optimization)")
            elif 'meta' in script:
                print(f"   - scripts/{script} (Pure metaschedule)")
            else:
                print(f"   - scripts/{script}")
    print()
    
    # 7. Example commands
    print("7. Quick Start Commands:")
    print()
    print("Run ResNet18 metaschedule benchmark:")
    print("   python3 benchmarks/models_onnx.py -m meta -a x86 -t 1000 \\")
    print("           -l results/my_resnet18_test -b models/resnet18.onnx")
    print()
    print("Run DPMeta hybrid optimization (requires metaschedule baseline first):")
    print("   # Step 1: Generate metaschedule baseline")
    print("   python3 benchmarks/models_onnx.py -m meta -a x86 -t 10000 \\")
    print("           -l results/resnet18_meta_baseline -b models/resnet18.onnx")
    print("   # Step 2: Run DPMeta hybrid optimization")
    print("   python3 benchmarks/models_onnx.py -m dpmeta -a x86 -t 100 -k 1000 \\")
    print("           -l results/resnet18_meta_baseline -b models/resnet18.onnx")
    print()
    print("Run complete benchmark suite:")
    print("   ./scripts/meta_benchmarks.sh  # Pure metaschedule")
    print("   ./scripts/dpmeta_benchmarks.sh  # DPMeta hybrid optimization")
    print()
    print("Analyze existing results:")
    print("   cat results/ms/meta_cuda_a100_resnet18_10k/output.txt")
    print("   ls results/ms/meta_cuda_a100_resnet18_10k/logs/  # DPMeta layer logs if available")
    print()

def main():
    """Main function."""
    try:
        analyze_benchmark_results()
        print("=" * 60)
        print("DROPLET HYBRID OPTIMIZATION SUMMARY")
        print("=" * 60)
        print("Bennu supports three main optimization approaches:")
        print("1. Pure Metaschedule (-m meta): TVM's standard metaschedule tuning")
        print("2. DPMeta Hybrid (-m dpmeta): Droplet search + Metaschedule combination")
        print("3. Pure Droplet (-m droplet): Droplet search on Ansor results")
        print()
        print("DPMeta hybrid optimization combines the best of both worlds:")
        print("- Uses metaschedule's broad search capabilities for initial optimization")
        print("- Applies Droplet's focused search for fine-tuning top results")
        print("- Often achieves better performance than either method alone")
        print()
        print("For detailed documentation, see METASCHEDULE_BENCHMARKS.md")
        print("For interactive demo, run: ./run_resnet_benchmarks.sh")
    except Exception as e:
        print(f"Error analyzing benchmarks: {e}")

if __name__ == "__main__":
    main()