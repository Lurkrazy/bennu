import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def find_best_gflops_col(df):
    for col in df.columns:
        if 'best_gflops' == col.lower():
            return col
    for col in df.columns:
        if 'gflop' in col.lower():
            return col
    return None

out_dir = "analyze_gflops_convergence_output"
os.makedirs(out_dir, exist_ok=True)

FILES = sorted(glob.glob("./layer_*/parsed_trials.csv"))

for path in FILES:
    layer_name = os.path.basename(os.path.dirname(path))
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        continue
    col = find_best_gflops_col(df)
    if col is None:
        print(f"No gflops column found in {path}. Available columns: {list(df.columns)}")
        continue
    cleaned = df[col].astype(str).str.strip().str.rstrip('.')
    series = pd.to_numeric(cleaned, errors='coerce').dropna()
    if series.empty:
        print(f"No numeric gflops values in {path}.")
        continue

    # Scatter plot for distribution (use gflops column)
    gflops_col = None
    for colname in df.columns:
        if 'gflop' in colname.lower() and 'best' not in colname.lower():
            gflops_col = colname
            break
    if gflops_col is not None:
        gflops_cleaned = df[gflops_col].astype(str).str.strip().str.rstrip('.')
        gflops_series = pd.to_numeric(gflops_cleaned, errors='coerce').dropna()
        if not gflops_series.empty:
            plt.figure(figsize=(8, 5))
            plt.scatter(range(len(gflops_series)), gflops_series.values, alpha=0.7, s=18)
            plt.title(f"{layer_name} gflops Scatter Plot")
            plt.xlabel("Trial")
            plt.ylabel("gflops")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{layer_name}_gflops_scatter.png"))
            plt.close()

    # Convergence plot (trial order, use best_gflops)
    plt.figure(figsize=(8, 5))
    plt.plot(series.values, marker='o', linestyle='-', alpha=0.7)
    plt.title(f"{layer_name} best_gflops Convergence")
    plt.xlabel("Trial")
    plt.ylabel("best_gflops")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{layer_name}_best_gflops_convergence.png"))
    plt.close()