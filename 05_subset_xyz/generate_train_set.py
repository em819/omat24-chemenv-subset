#!/usr/bin/env python3
"""
Materialise a training set: given a file of OMat24 mat_ids (one per line), scan
the source extxyz files and write every matching structure into a single output
xyz. No DuckDB, no frequency matrix, no train/val split -- just "mat_ids -> one xyz".

Use it to turn each chemenv-coverage subset (or random/coreset baseline) into a
train.xyz. Validation is the official OMat24 validation split (shared across all
models), so we do NOT split here -- the whole subset is the training set.

HOW THE MATCHING WORKS (identical to the original):
  Each extxyz frame is
      <n_atoms>\n
      <comment line>      # carries 'sid=<mat_id>' among space-separated tokens
      <n_atoms atom lines>
  with no blank separators between frames. We stream frame-by-frame, pull the
  'sid=' token from the comment, and if that sid is in the requested set we copy
  the frame's RAW TEXT verbatim to the output (no parsing -> exact round-trip).
  A mat_id may appear in several frames (multiple configurations); ALL of its
  frames are written. Frames whose sid is not requested are skipped.

USAGE
  python generate_train_set.py \
      --subset-mat-ids best_subset_from_chemenvs_freq_ge_100.txt \
      --xyz-dir  /.../OMat24_data/.../all_subsets \
      --output   subset_train.xyz
"""

import argparse
import glob
import os
import time

import numpy as np


def parse_sid(comment: str):
    """Return the mat_id from a comment line's 'sid=<...>' token, or None."""
    for token in comment.split():
        if token.startswith("sid="):
            return token[4:]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-mat-ids", required=True,
                    help="Text file of OMat24 mat_ids to include (one per line).")
    ap.add_argument("--xyz-dir", required=True,
                    help="Root directory holding the source extxyz files.")
    ap.add_argument("--output", required=True,
                    help="Output xyz path (all matching structures concatenated).")
    ap.add_argument("--glob-pattern", default="**/*xyz",
                    help="Glob (relative to --xyz-dir) for source xyz files, "
                         "matched recursively. The default finds both "
                         "<xyz-dir>/<subset>/*.xyz and <xyz-dir>/<subset>.xyz.")
    ap.add_argument("--report-every", type=int, default=500_000,
                    help="Print progress every N frames scanned.")
    ap.add_argument("--strict-sid", action="store_true",
                    help="Abort if a frame's comment line has no 'sid=' token "
                         "(default: warn once and skip such frames).")
    args = ap.parse_args()

    t0 = time.time()

    # ---- requested mat_ids -> set (atleast_1d guards the single-line case) ----
    wanted = set(map(str, np.atleast_1d(
        np.genfromtxt(args.subset_mat_ids, dtype="str"))))
    print(f"Requested {len(wanted):,} unique mat_ids from {args.subset_mat_ids}",
          flush=True)

    xyz_files = sorted(glob.glob(os.path.join(args.xyz_dir, args.glob_pattern),
                                 recursive=True))
    if not xyz_files:
        raise SystemExit(f"no source xyz files matched "
                         f"{os.path.join(args.xyz_dir, args.glob_pattern)}")
    print(f"Found {len(xyz_files)} source xyz file(s).", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    n_scanned = 0          # total frames seen
    n_written = 0          # frames copied (== requested structures, incl. repeats)
    found = set()          # mat_ids matched at least once
    warned_missing_sid = False

    with open(args.output, "w") as out:
        for xyz_path in xyz_files:
            print(f"  Reading {xyz_path} ...", flush=True)
            with open(xyz_path, "r") as f:
                while True:
                    line1 = f.readline()
                    if not line1:
                        break                      # EOF
                    n_atoms = int(line1.strip())
                    comment = f.readline()
                    atom_lines = [f.readline() for _ in range(n_atoms)]
                    n_scanned += 1
                    if args.report_every and n_scanned % args.report_every == 0:
                        print(f"    {n_scanned:,} frames scanned, "
                              f"{n_written:,} written ...", flush=True)

                    sid = parse_sid(comment)
                    if sid is None:
                        if args.strict_sid:
                            raise ValueError(
                                f"No 'sid=' in comment of frame {n_scanned} in "
                                f"{xyz_path}.\nComment: {comment}")
                        if not warned_missing_sid:
                            print(f"    WARNING: frame {n_scanned} in {xyz_path} has "
                                  f"no 'sid=' token; skipping such frames.", flush=True)
                            warned_missing_sid = True
                        continue

                    if sid not in wanted:
                        continue                   # not in our subset

                    out.write(line1 + comment + "".join(atom_lines))
                    n_written += 1
                    found.add(sid)

    missing = wanted - found
    print(f"\nDone in {(time.time()-t0)/60:.2f} min.")
    print(f"  frames scanned : {n_scanned:,}")
    print(f"  frames written : {n_written:,}  -> {args.output}")
    print(f"  mat_ids matched: {len(found):,} / {len(wanted):,} requested")
    if missing:
        print(f"  WARNING: {len(missing):,} requested mat_ids were not found in any "
              f"source xyz. Examples:")
        for mid in list(missing)[:5]:
            print(f"    {mid}")


if __name__ == "__main__":
    main()
