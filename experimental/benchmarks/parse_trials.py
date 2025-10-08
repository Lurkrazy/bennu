#!/usr/bin/env python3
import glob
import os
import re
import csv
import sys

TRIAL_RE = re.compile(r'Trial\s*#\s*(\d+)', re.IGNORECASE)
GFLOPS_RE = re.compile(r'GFLOPs:\s*([0-9eE\+\-\,\.]+)')
TIME_RE = re.compile(r'Time:\s*([0-9eE\+\-\,\.]+)\s*us', re.IGNORECASE)
BEST_GFLOPS_RE = re.compile(r'Best\s+GFLOPs:\s*([0-9eE\+\-\,\.]+)', re.IGNORECASE)

def clean_number(s):
    if s is None:
        return ''
    s = s.replace(',', '')
    return s

def parse_log(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [l.rstrip('\n') for l in f]
    except Exception as e:
        print(f"Skipping {path}: failed to read ({e})", file=sys.stderr)
        return []

    rows = []
    n = len(lines)
    for i, line in enumerate(lines):
        m = TRIAL_RE.search(line)
        if not m:
            continue
        idx = int(m.group(1))
        gflops = None
        time_us = None
        best_gflops = None

        # Try same line first, then up to next 2 lines
        snippet = ' '.join(lines[i:i+3])
        m_g = GFLOPS_RE.search(snippet)
        if m_g:
            gflops = clean_number(m_g.group(1))
        m_t = TIME_RE.search(snippet)
        if m_t:
            time_us = clean_number(m_t.group(1))
        m_b = BEST_GFLOPS_RE.search(snippet)
        if m_b:
            best_gflops = clean_number(m_b.group(1))

        rows.append({
            'trial_index': idx,
            'gflops': gflops or '',
            'time_us': time_us or '',
            'best_gflops': best_gflops or '',
        })

    return rows

def write_csv(layer_dir, rows):
    out_path = os.path.join(layer_dir, 'parsed_trials.csv')
    try:
        with open(out_path, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['trial_index','gflops','time_us','best_gflops'])
            # sort by trial_index to keep deterministic order
            for r in sorted(rows, key=lambda x: x['trial_index']):
                writer.writerow([r['trial_index'], r['gflops'], r['time_us'], r['best_gflops']])
    except Exception as e:
        print(f"Failed to write {out_path}: {e}", file=sys.stderr)
        return False
    return True

def main():
    pattern = 'layer_*/logs/tvm.meta_schedule.logging.task_0_main.log'
    log_paths = sorted(glob.glob(pattern))
    if not log_paths:
        print("No log files found.", file=sys.stderr)
        return

    layers = 0
    total_trials = 0
    written_layers = []
    for p in log_paths:
        layer_dir = os.path.dirname(os.path.dirname(p))
        rows = parse_log(p)
        try:
            # Ensure we write a CSV even if no trials found (empty with header)
            ok = write_csv(layer_dir, rows)
            if ok:
                layers += 1
                total_trials += len(rows)
                written_layers.append(os.path.join(layer_dir, 'parsed_trials.csv'))
        except Exception as e:
            print(f"Error processing {p}: {e}", file=sys.stderr)
            continue

    print(f"Processed {layers} layers, parsed {total_trials} total trials.")
    for w in written_layers:
        print(f"Wrote {w}")

if __name__ == '__main__':
    main()