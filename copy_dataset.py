#!/usr/bin/env python3
"""
copy_dataset.py - Copy one T2 + mask pair per subject to an output directory.

Usage:
    python3 copy_dataset.py [t2s.txt] [masks.txt] [output_dir] [--jobs N]
    ./copy_dataset.py t2s.txt masks.txt data/ --jobs 8

Default jobs = min(8, cpu_count).  Tune upward for fast NAS/NVMe, downward
for spinning drives where parallel seeks hurt.

Subdir priority (highest to lowest):
    1. T2reg_SD
    2. T2reg
    3. Date-code dirs (e.g. 240520bd, 250321ts) — highest numeric date wins
    4. Top-level Processed/ (file directly in Processed/, no subdir)
    5. T2reg_images
    6. Anything else (Junk, NoERC, nested subdirs like T2reg_SD/bad, etc.)

Only copies a subject when BOTH a T2 and a mask exist under the same subdir.
Output: data/{SUBJECT_ID}.nii.gz  and  data/{SUBJECT_ID}_mask.nii.gz
"""

import sys
import re
import shutil
import os
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Progress bar (stdlib only) ────────────────────────────────────────────────────
class Progress:
    BAR_WIDTH = 40

    def __init__(self, total: int):
        self.total   = total
        self.done    = 0
        self.start   = time.monotonic()
        self._tty    = sys.stderr.isatty()

    def update(self, n: int = 1):
        self.done += n
        self._render()

    def _render(self):
        done, total = self.done, self.total
        frac = done / total if total else 1.0
        filled = int(self.BAR_WIDTH * frac)
        bar = '#' * filled + '-' * (self.BAR_WIDTH - filled)
        elapsed = time.monotonic() - self.start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        line = f"  [{bar}] {done}/{total}  {rate:.1f}/s  ETA {eta:.0f}s"
        if self._tty:
            print(f"\r{line}", end='', flush=True, file=sys.stderr)
        # non-tty: only print at 100%
        if done == total:
            print(file=sys.stderr)  # newline after bar

# ── Priority assignment ───────────────────────────────────────────────────────
DATE_CODE_RE = re.compile(r'^[0-9]{6}[a-zA-Z]*$')

def priority(subdir: str) -> tuple:
    """Return a sort key (lower = higher priority).
    For date-code dirs, secondary key is the numeric part (negated so higher
    date = better when sorted ascending).
    """
    if subdir == "T2reg_SD":
        return (1, 0)
    if subdir == "T2reg":
        return (2, 0)
    if subdir == "" :          # directly in Processed/
        return (4, 0)
    if subdir == "T2reg_images":
        return (5, 0)
    if DATE_CODE_RE.match(subdir):
        # Extract the 6-digit date number; negate so largest sorts first
        num = int(re.sub(r'[^0-9]', '', subdir))
        return (3, -num)
    return (6, 0)

# ── Path parsing ──────────────────────────────────────────────────────────────
SUBJECT_RE = re.compile(r'/Prostate_data/([0-9]+)/')

def parse_path(path: str):
    """Return (subject_id, subdir) from a data path."""
    m = SUBJECT_RE.search(path)
    if not m:
        return None, None
    subject = m.group(1)

    # Everything after .../Processed/
    processed_idx = path.find('/Processed/')
    if processed_idx == -1:
        return subject, ""
    after = path[processed_idx + len('/Processed/'):]

    filename = Path(path).name
    subdir = after[: -(len(filename) + 1)] if after.endswith('/' + filename) else ""
    return subject, subdir

# ── Load file into dicts ─────────────────────────────────────────────────────
# Returns:
#   flat:    {(subject, subdir): path}
#   by_subj: {subject: {subdir: path}}
def load_file(filepath: str) -> tuple:
    flat = {}
    by_subj: dict[str, dict] = defaultdict(dict)
    with open(filepath) as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            subject, subdir = parse_path(path)
            if subject is None:
                print(f"  WARNING: cannot parse subject from: {path}", file=sys.stderr)
                continue
            flat[(subject, subdir)] = path
            by_subj[subject][subdir] = path
    return flat, by_subj

# ── Main ──────────────────────────────────────────────────────────────────────
def _copy_subject(subj, t2_src, mask_src, t2_dest, mask_dest, label):
    """Copy one subject's pair; returns (subj, label, error_msg|None)."""
    try:
        shutil.copy2(t2_src, t2_dest)
        shutil.copy2(mask_src, mask_dest)
        return (subj, label, None)
    except OSError as e:
        return (subj, label, str(e))


def main():
    args = sys.argv[1:]

    # Parse optional --jobs / -j flag
    jobs = min(8, os.cpu_count() or 4)
    for flag in ('-j', '--jobs'):
        if flag in args:
            idx = args.index(flag)
            try:
                jobs = int(args[idx + 1])
                args = args[:idx] + args[idx + 2:]
            except (IndexError, ValueError):
                print(f"ERROR: {flag} requires an integer argument", file=sys.stderr)
                sys.exit(1)

    t2_file    = args[0] if len(args) > 0 else "t2s.txt"
    mask_file  = args[1] if len(args) > 1 else "masks.txt"
    output_dir = Path(args[2] if len(args) > 2 else "data")

    for f in [t2_file, mask_file]:
        if not Path(f).is_file():
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    _, t2_by_subj = load_file(t2_file)
    mask_flat, _  = load_file(mask_file)

    all_subjects = sorted(t2_by_subj, key=lambda x: int(x))

    # Build work list and handle no-pair subjects up front
    work = []      # [(subj, t2_src, mask_src, t2_dest, mask_dest, label)]
    no_pair = 0

    for subj in all_subjects:
        paired_subdirs = [
            subdir
            for subdir in t2_by_subj[subj]
            if (subj, subdir) in mask_flat
        ]
        if not paired_subdirs:
            print(f"SKIP    {subj:<6}  no paired T2+mask in same subdir")
            no_pair += 1
            continue

        best_subdir = min(paired_subdirs, key=priority)
        label       = best_subdir if best_subdir else "Processed/"
        work.append((
            subj,
            t2_by_subj[subj][best_subdir],
            mask_flat[(subj, best_subdir)],
            output_dir / f"{subj}.nii.gz",
            output_dir / f"{subj}_mask.nii.gz",
            label,
        ))

    # Parallel copy
    copied = 0
    errors = 0
    bar = Progress(len(work))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_copy_subject, *item): item[0]
            for item in work
        }
        for fut in as_completed(futures):
            subj, label, err = fut.result()
            bar.update()
            if err:
                print(f"\rERROR   {subj:<6}  {err}", file=sys.stderr)
                errors += 1
            else:
                print(f"\rCopied  {subj:<6}  [{label}]")
                copied += 1

    print(f"\nDone: {copied} copied, {no_pair} no-pair skipped, {errors} errors")

if __name__ == "__main__":
    main()
