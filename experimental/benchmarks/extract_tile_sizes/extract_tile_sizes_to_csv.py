#!/usr/bin/env python3
import os
import json
import csv

# This script extracts per-trial tile sizes and execution times from each
# layer's database_tuning_record.json and writes:
#  - per-layer CSV: layer_X/tile_sizes.csv
#  - combined CSV: agents_work/agents_kilocode_extract_tile_sizes_20251009/all_layers_tile_sizes.csv
#
# CSV columns: layer, trial_index, tile_size (JSON string), execution_time (JSON string)

# locate project root (two levels up from this script: /workspace/.../benchmarks)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

layer_dirs = [f"layer_{i}" for i in range(10)]
json_filename = "database_tuning_record.json"
per_layer_csv_name = "tile_sizes.csv"
combined_csv_path = os.path.join(SCRIPT_DIR, "all_layers_tile_sizes.csv")

combined_rows = []

for layer in layer_dirs:
    json_path = os.path.join(PROJECT_ROOT, layer, json_filename)
    csv_path = os.path.join(PROJECT_ROOT, layer, per_layer_csv_name)

    if not os.path.exists(json_path):
        # skip missing layers
        continue

    rows = []
    with open(json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
            except Exception:
                # skip malformed lines
                continue

            # data structure assumed: [some_meta, params, ...] where params contains tile sizes & time
            # original scripts used: params = data[1]; tile_size = params[0][-1]; exec_time = params[1]
            params = data[1] if len(data) > 1 else None

            tile_size = None
            exec_time = None

            if params is not None:
                try:
                    tile_size = params[0][-1]
                except Exception:
                    tile_size = None
                try:
                    exec_time = params[1]
                except Exception:
                    exec_time = None

            row = {
                "layer": layer,
                "trial_index": idx,
                "tile_size": tile_size,
                "execution_time": exec_time,
            }
            rows.append(row)
            combined_rows.append(row)

    # write per-layer CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "trial_index", "tile_size", "execution_time"])
        for row in rows:
            writer.writerow([
                row["layer"],
                row["trial_index"],
                json.dumps(row["tile_size"], ensure_ascii=False),
                json.dumps(row["execution_time"], ensure_ascii=False),
            ])

# write combined CSV (if any rows)
if combined_rows:
    with open(combined_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "trial_index", "tile_size", "execution_time"])
        for row in combined_rows:
            writer.writerow([
                row["layer"],
                row["trial_index"],
                json.dumps(row["tile_size"], ensure_ascii=False),
                json.dumps(row["execution_time"], ensure_ascii=False),
            ])

print(f"Wrote per-layer CSVs to each layer directory (file: {per_layer_csv_name}) and combined CSV to: {combined_csv_path}")
