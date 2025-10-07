#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract SamplePerfectTile + Split + (decision vectors) from a TVM MetaSchedule trace string.

Usage:
  python extract_tiles.py < trace.txt
  # or
  python extract_tiles.py path/to/trace.txt

Outputs a structured summary for each tiling site:
- loop id (e.g., l26)
- requested levels (n_levels, max_innermost_factor) from SamplePerfectTile
- temp tile vars (vXX list)
- Split-produced loops (lXX list)
- chosen factors (if found in the decisions list)
"""

import sys
import re
import json
from typing import List, Dict, Optional, Tuple

SAMPLE_TILE_RE = re.compile(
    r'\["SamplePerfectTile",\s*\["(?P<loop>l\d+)"\]\s*,\s*\['
    r'(?P<n_levels>\d+)\s*,\s*(?P<max_inn>\d+)'
    r'\]\s*,\s*\[(?P<vars>(?:"v\d+"\s*(?:,\s*"v\d+"\s*)*))\]\s*\]'
)

SPLIT_RE = re.compile(
    r'\["Split",\s*\["(?P<loop>l\d+)"\s*,\s*(?P<vars>\[(?:[^][]|\[[^][]*\])*\])\s*\]\s*,\s*\[1\s*,\s*0\]\s*,\s*\['
    r'(?P<outs>(?:"l\d+"\s*(?:,\s*"l\d+"\s*)*))\]\s*\]'
)

# Decision tuples look like: [31,[49,16,1,1,1]]
# We'll match many of them; later we’ll map in occurrence order to SamplePerfectTile sites.
DECISION_PAIR_RE = re.compile(
    r'\[\s*(?P<step>\d+)\s*,\s*\[(?P<vector>(?:-?\d+\s*(?:,\s*-?\d+\s*)*))\]\s*\]'
)

def _split_csv_strings(s: str) -> List[str]:
    """Split a JSON-like CSV of quoted strings: "v1","v2","v3" -> ['v1','v2','v3']"""
    return [x.strip().strip('"') for x in s.split(',') if x.strip()]

def _split_csv_ints(s: str) -> List[int]:
    """Split a CSV of ints: '1, 2, 3' -> [1,2,3]"""
    return [int(x.strip()) for x in s.split(',') if x.strip()]

def parse_sample_tiles(text: str) -> List[Dict]:
    tiles = []
    for m in SAMPLE_TILE_RE.finditer(text):
        tiles.append({
            "loop": m.group("loop"),
            "n_levels": int(m.group("n_levels")),
            "max_innermost_factor": int(m.group("max_inn")),
            "tile_vars": _split_csv_strings(m.group("vars")),
            # Will be filled later if we find the matching Split/decisions
            "split_out_loops": None,
            "chosen_factors": None,
            "decision_step": None,
        })
    return tiles

def parse_splits(text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Return mapping: loop_id -> { 'vars': ['v..'], 'outs': ['l..'] }

    The "Split" trace can appear in multiple formats:
      - ["Split",["l26","v29","v30"...],[1,0],["l34",...]]
      - ["Split",["l26",[...vars...]],...]
    This implementation uses a looser regex to capture both styles.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    pattern = re.compile(
        r'\["Split",\s*\[(?P<items>[^\]]+)\]\s*,\s*\[[^\]]*\]\s*,\s*\[(?P<outs>(?:"l\d+"\s*(?:,\s*"l\d+"\s*)*))\]\s*\]'
    )
    for m in pattern.finditer(text):
        items = m.group("items")
        # find the first loop id like "l26"
        loop_m = re.search(r'"(l\d+)"', items)
        if not loop_m:
            continue
        loop = loop_m.group(1)
        # collect any vNN names present in the items sequence
        vnames = re.findall(r'"(v\d+)"', items)
        outs = _split_csv_strings(m.group("outs"))
        out[loop] = {"vars": vnames, "outs": outs}
    return out

def parse_decisions(text: str) -> List[Tuple[int, List[int]]]:
    """
    Returns a list of (step_id, vector) in the order they appear.
    Scan the entire trace text for decision pairs like [31,[49,16,1,1,1]].
    """
    pairs: List[Tuple[int, List[int]]] = []
    for m in DECISION_PAIR_RE.finditer(text):
        vec = _split_csv_ints(m.group("vector"))
        pairs.append((int(m.group("step")), vec))
    return pairs

def attach_splits(tiles: List[Dict], split_map: Dict[str, Dict[str, List[str]]]) -> None:
    for t in tiles:
        info = split_map.get(t["loop"])
        if info:
            # sanity check: tile_vars and split vars should align
            if info["vars"] and t["tile_vars"]:
                # The Split vars are usually the SamplePerfectTile outputs (vXX...).
                # Not all logs guarantee exact equality, but we record regardless.
                t["split_out_loops"] = info["outs"]
            else:
                t["split_out_loops"] = info["outs"]

def attach_decisions_in_order(tiles: List[Dict], decision_pairs: List[Tuple[int, List[int]]]) -> List[str]:
    """
    Attach decision vectors to tiles by order of appearance:
      first SamplePerfectTile <- first vector in decisions,
      second <- second, etc.
    Returns a list of warnings (if any).
    """
    warnings = []
    if not decision_pairs:
        warnings.append("No decision vectors found; returning tiles without chosen factors.")
        return warnings

    # Heuristic: keep only vectors with length >= 3 and <= 6 (typical perfect tile lengths).
    filtered = [(s, v) for (s, v) in decision_pairs if 1 <= len(v) <= 8]
    if len(filtered) < len(tiles):
        warnings.append(
            f"Found only {len(filtered)} decision vector(s) for {len(tiles)} SamplePerfectTile site(s). "
            "Mapping by order for available ones."
        )

    for i, tile in enumerate(tiles):
        if i < len(filtered):
            step, vec = filtered[i]
            tile["chosen_factors"] = vec
            tile["decision_step"] = step
        else:
            tile["chosen_factors"] = None
            tile["decision_step"] = None
    return warnings

def extract_all(text: str) -> Dict:
    tiles = parse_sample_tiles(text)
    split_map = parse_splits(text)
    attach_splits(tiles, split_map)
    decisions = parse_decisions(text)
    warnings = attach_decisions_in_order(tiles, decisions)

    return {
        "num_tiles": len(tiles),
        "tiles": tiles,
        "warnings": warnings,
    }


def main():
    
    # read from file
    
    json_path = "/workspace/tvm-dev/bennu/experimental/extractor/data/ms/layer_mini_0/database_tuning_record.json"
    
    
    # read entire file (not only the first line) so regex can match across lines
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    result = extract_all(text)
    # Pretty print JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
