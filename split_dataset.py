#!/usr/bin/env python3
"""
split_dataset.py - Stratified train/val/test split balanced across scanner types.

The first digit of the subject ID identifies the scanner type.
Each scanner type is split independently so the overall class ratios are
preserved in every subset.

Usage:
    python3 split_dataset.py [data_dir] [--train F] [--val F] [--test F]
                             [--seed N] [--out-dir DIR] [--format FORMAT]

Arguments:
    data_dir        Directory containing {SUBJECT_ID}.nii.gz files (default: data/)
    --train F       Fraction for training set   (default: 0.70)
    --val   F       Fraction for validation set (default: 0.15)
    --test  F       Fraction for test set       (default: 0.15)
    --seed  N       Random seed for reproducibility (default: 42)
    --out-dir DIR   Where to write split files (default: same as data_dir)
    --format FORMAT Output format: 'txt' (one ID per line) or 'csv' (default: txt)

Outputs (in out-dir):
    train.txt / val.txt / test.txt  — subject IDs, one per line

Example:
    python3 split_dataset.py data/ --train 0.8 --val 0.1 --test 0.1
"""

import sys
import os
import re
import random
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("data_dir", nargs="?", default="data",
                   help="Directory of copied NIfTI files (default: data/)")
    p.add_argument("--train", type=float, default=0.70, metavar="F")
    p.add_argument("--val",   type=float, default=0.20, metavar="F")
    p.add_argument("--test",  type=float, default=0.10, metavar="F")
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="Output directory for split files (default: data_dir)")
    p.add_argument("--format", choices=["txt", "csv", "json"], default="json")
    args = p.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        p.error(f"--train + --val + --test must sum to 1.0 (got {total:.4f})")

    return args

# ── Discover subjects ─────────────────────────────────────────────────────────
def find_subjects(data_dir: Path) -> list[str]:
    """Return sorted list of subject IDs found as {ID}.nii.gz in data_dir."""
    ids = []
    for p in data_dir.iterdir():
        # Match NNNN.nii.gz but NOT NNNN_mask.nii.gz
        m = re.fullmatch(r'([0-9]+)\.nii\.gz', p.name)
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=lambda x: int(x))

# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(subjects: list[str], train_f: float, val_f: float,
                     seed: int) -> tuple[list, list, list]:
    """Split each scanner-type group proportionally, then combine."""
    rng = random.Random(seed)

    # Group by first digit
    groups: dict[str, list[str]] = defaultdict(list)
    for subj in subjects:
        groups[subj[0]].append(subj)

    train, val, test = [], [], []

    for scanner, group in sorted(groups.items()):
        g = group.copy()
        rng.shuffle(g)
        n = len(g)
        n_train = round(n * train_f)
        n_val   = round(n * val_f)
        # test gets the remainder to avoid rounding loss
        train += g[:n_train]
        val   += g[n_train:n_train + n_val]
        test  += g[n_train + n_val:]

    # Shuffle the combined lists so scanner types are interleaved
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test

# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(train, val, test):
    all_subj = train + val + test
    scanners = sorted({s[0] for s in all_subj})

    col_w = 8
    header = f"  {'Scanner':<10}" + "".join(f"{'Train':>{col_w}}{'Val':>{col_w}}{'Test':>{col_w}}{'Total':>{col_w}}")
    print()
    print(header)
    print("  " + "-" * (10 + col_w * 4))

    for sc in scanners:
        tr = sum(1 for s in train if s[0] == sc)
        va = sum(1 for s in val   if s[0] == sc)
        te = sum(1 for s in test  if s[0] == sc)
        tot = tr + va + te
        print(f"  {sc:<10}{tr:>{col_w}}{va:>{col_w}}{te:>{col_w}}{tot:>{col_w}}")

    print("  " + "-" * (10 + col_w * 4))
    print(f"  {'Total':<10}{len(train):>{col_w}}{len(val):>{col_w}}{len(test):>{col_w}}{len(all_subj):>{col_w}}")

    total = len(all_subj)
    pct = lambda n: f"{100*n/total:.1f}%" if total else "-"
    print(f"  {'%':<10}{pct(len(train)):>{col_w}}{pct(len(val)):>{col_w}}{pct(len(test)):>{col_w}}")
    print()

# ── Write output ──────────────────────────────────────────────────────────────
def write_splits_json(out_dir: Path, splits: dict[str, list[str]]) -> Path:
    path = out_dir / "dataset.json"
    data = {
        split: [{"image": f"{s}.nii.gz", "label": f"{s}_mask.nii.gz"} for s in subjects]
        for split, subjects in splits.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def write_split(out_dir: Path, name: str, subjects: list[str], fmt: str):
    ext = "txt" if fmt == "txt" else "csv"
    path = out_dir / f"{name}.{ext}"
    with open(path, "w") as f:
        if fmt == "csv":
            f.write("subject_id,scanner\n")
            for s in subjects:
                f.write(f"{s},{s[0]}\n")
        else:
            for s in subjects:
                f.write(s + "\n")
    return path

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(data_dir)
    if not subjects:
        print(f"ERROR: no subject NIfTI files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(subjects)} subjects  |  "
          f"seed={args.seed}  |  "
          f"split {args.train:.0%}/{args.val:.0%}/{args.test:.0%}")

    train, val, test = stratified_split(subjects, args.train, args.val, args.seed)

    print_summary(train, val, test)

    splits = {"train": train, "val": val, "test": test}
    if args.format == "json":
        p = write_splits_json(out_dir, splits)
        print(f"  Wrote splits → {p}")
    else:
        for name, subjs in splits.items():
            p = write_split(out_dir, name, subjs, args.format)
            print(f"  Wrote {name:<6} → {p}  ({len(subjs)} subjects)")
    print()

if __name__ == "__main__":
    main()
