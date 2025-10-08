#!/usr/bin/env python3
"""Plot best_gflops vs trials for each layer's parsed_trials.csv."""
import os
import glob
import csv
import math
import sys
from typing import List, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib is required to run this script. Install it via pip.", file=sys.stderr)
    sys.exit(1)


def parse_number(s):
    if s is None:
        return None
    s = s.strip()
    if s == '':
        return None
    try:
        if '.' in s or 'e' in s or 'E' in s:
            return float(s)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def process_csv(path: str) -> Tuple[List[float], List[float]]:
    x = []
    y = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                tx = parse_number(row.get('trial_index'))
                if tx is None:
                    print(f"warning: missing trial_index at row {i+1} in {path}", file=sys.stderr)
                    continue
                best = row.get('best_gflops')
                by = parse_number(best)
                if by is None or (isinstance(by, float) and math.isnan(by)):
                    # treat missing as NaN and skip
                    continue
                x.append(float(tx))
                y.append(float(by))
            except Exception as e:
                print(f"warning: failed to parse row {i+1} in {path}: {e}", file=sys.stderr)
                continue
    return x, y


def layer_name_from_path(path: str) -> str:
    # path like 'layer_0/parsed_trials.csv'
    dirn = os.path.dirname(path)
    name = os.path.basename(dirn)
    return name


def main():
    pattern = os.path.join('layer_*', 'parsed_trials.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print("No layer CSVs found.", file=sys.stderr)
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    images_written = []
    processed = 0
    for path in files:
        processed += 1
        try:
            x, y = process_csv(path)
        except FileNotFoundError:
            print(f"warning: unreadable file {path}, skipping", file=sys.stderr)
            continue
        except Exception as e:
            print(f"warning: failed to read {path}: {e}", file=sys.stderr)
            continue
        layer = layer_name_from_path(path)
        img_name = f"{layer}_best_gflops.png"
        out_path = os.path.join(output_dir, img_name)
        if not x or not y:
            print(f"warning: no valid best_gflops data for {path}, skipping image generation", file=sys.stderr)
            continue
        try:
            plt.figure()
            plt.plot(x, y, marker='o', linestyle='-')
            plt.title(f"{layer} best_gflops vs trials")
            plt.xlabel("trial index")
            plt.ylabel("best_gflops")
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            images_written.append(out_path)
        except Exception as e:
            print(f"warning: failed to write image for {path}: {e}", file=sys.stderr)
            continue

    print(f"Processed {processed} layer CSVs, wrote {len(images_written)} images.")
    if images_written:
        for p in images_written:
            print(p)
    # exit with 0


if __name__ == '__main__':
    main()