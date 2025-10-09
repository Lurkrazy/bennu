import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def find_gflops_col(df):
    for col in df.columns:
        if 'gflop' in col.lower():
            return col
    return None

FILES = [
    './layer_0/parsed_trials.csv',
    './layer_1/parsed_trials.csv',
    './layer_2/parsed_trials.csv',
    './layer_3/parsed_trials.csv',
    './layer_4/parsed_trials.csv',
    './layer_5/parsed_trials.csv',
    './layer_6/parsed_trials.csv',
    './layer_7/parsed_trials.csv',
    './layer_8/parsed_trials.csv',
    './layer_9/parsed_trials.csv',
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir
    os.makedirs(out_dir, exist_ok=True)

    stats_rows = []
    combined = []

    for path in FILES:
        layer_name = os.path.basename(os.path.dirname(path))
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f'Failed to read {path}: {e}', file=sys.stderr)
            continue
        col = find_gflops_col(df)
        if col is None:
            print(f'No gflops column found in {path}', file=sys.stderr)
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            print(f'No numeric gflops values in {path}', file=sys.stderr)
            continue
        combined.append(pd.DataFrame({'layer': layer_name, 'gflops': series.values}))

        stats_rows.append({
            'layer': layer_name,
            'count': int(series.count()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
        })

        plt.figure(figsize=(6,4))
        sns.histplot(series, bins=40, kde=True)
        plt.title(f'{layer_name} gflops')
        plt.xlabel('GFLOPS')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{layer_name}_gflops_hist.png'))
        plt.close()

    if not stats_rows:
        print('No data processed.', file=sys.stderr)
        return

    stats_df = pd.DataFrame(stats_rows).sort_values('layer')
    stats_df.to_csv(os.path.join(out_dir, 'gflops_stats_summary.csv'), index=False)

    all_df = pd.concat(combined, ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, 'gflops_combined.csv'), index=False)

    plt.figure(figsize=(8,6))
    sns.histplot(data=all_df, x='gflops', hue='layer', element='step', stat='density', common_norm=False, bins=80, alpha=0.4)
    plt.title('Combined GFLOPS distribution by layer')
    plt.xlabel('GFLOPS')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_gflops_distribution.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    for name, group in all_df.groupby('layer'):
        sns.kdeplot(group['gflops'], label=name, linewidth=1.5)
    plt.legend(title='layer')
    plt.title('GFLOPS KDE by layer')
    plt.xlabel('GFLOPS')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_gflops_kde.png'))
    plt.close()

    print('Output written to', out_dir)

if __name__ == '__main__':
    main()