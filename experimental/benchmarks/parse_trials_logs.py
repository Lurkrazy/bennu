#!/usr/bin/env python3
"""Parse per-layer trial logs and write CSVs."""
import re
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

TRIAL_RE = re.compile(r"Trial #\s*(\d+)")
METRICS_RE = re.compile(
    r"GFLOPs:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\.?\s*Time:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*us\.?\s*Best GFLOPs:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

def parse_layer(layer_idx):
    # define paths for log and output CSV
    log_path = Path(f"layer_{layer_idx}") / "logs" / "tvm.meta_schedule.logging.task_0_main.log"
    csv_path = Path(f"layer_{layer_idx}") / "parsed_trials.csv"

    rows = []

    if not log_path.exists():
        logging.warning("Missing log file: %s", log_path)
        # still create CSV with header
        write_csv(csv_path, rows)
        return csv_path, 0

    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        logging.warning("Unable to read %s: %s", log_path, e)
        write_csv(csv_path, rows)
        return csv_path, 0

    for i, line in enumerate(lines):
        # fast check for trial marker
        if "Trial #" not in line:
            continue
        m_trial = TRIAL_RE.search(line)
        if not m_trial:
            continue
        trial_idx = int(m_trial.group(1))

        # search the same line and up to 3 following lines for metrics
        found = False
        for j in range(0, 4):
            k = i + j
            if k >= len(lines):
                break
            text = lines[k]
            m = METRICS_RE.search(text)
            if m:
                try:
                    gflops = float(m.group(1))
                    time_us = float(m.group(2))
                    best_gflops = float(m.group(3))
                except ValueError:
                    # skip this trial if conversion fails
                    break
                rows.append((trial_idx, gflops, time_us, best_gflops))
                found = True
                break
        if not found:
            # skip trials without parsable metrics
            continue

    write_csv(csv_path, rows)
    return csv_path, len(rows)

def write_csv(path, rows):
    # ensure parent dir exists and write CSV with header
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_index", "gflops", "time_us", "best_gflops"])
            for r in rows:
                writer.writerow(r)
    except Exception as e:
        logging.warning("Failed to write CSV %s: %s", path, e)

def main():
    summaries = []
    for idx in range(10):
        csv_path, count = parse_layer(idx)
        summaries.append((idx, csv_path, count))

    # concise summary to stdout
    for idx, csv_path, count in summaries:
        print(f"layer_{idx}/parsed_trials.csv: {count} trials")

if __name__ == "__main__":
    main()