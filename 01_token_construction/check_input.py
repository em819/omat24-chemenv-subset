#!/usr/bin/env python3
"""Preflight check for the stage-01 inputs. Run this BEFORE burning node-days.

Needs only ASE and numpy — not pymatgen, mpi4py, h5py or fairchem — so it runs
on a laptop or a login node in seconds.

It verifies, for each sub-dataset, that:

  * the sub-dataset resolves to exactly one of an ASE DB or an extxyz file
    (auto-detection refuses if both are present, since they may differ);
  * frame 0 parses;
  * frame 0 carries ``info['sid']``       -> becomes mat_id, and is the key
                                             stage 05 matches on;
  * frame 0 has a periodic cell           -> chemenv needs one;
  * the sid values look distinct across the first few frames.

Usage
-----
    # auto-detect ASE DB vs extxyz per sub-dataset, as the tokeniser does
    python check_input.py --data-dir /path/to/omat24

    python check_input.py --data-dir /path/to/omat24 --format extxyz
    python check_input.py --data-dir /path/to/omat24 \\
                          --file-template 'all_structures_{subset}.xyz'
    python check_input.py --data-dir /path/to/extracted --format ase-db

Exit status is 0 only if every sub-dataset passed.
"""

import argparse
import os
import sys

from input_backend import load_subsets, resolve_input

DEFAULT_SUBSETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "subsets.txt")


def first_frames_extxyz(path, n):
    import ase.io
    return ase.io.read(path, index=f":{n}", format="extxyz")


def first_frames_asedb(path, n):
    from fairchem.core.datasets import AseDBDataset
    ds = AseDBDataset(config=dict(src=path))
    return [ds.get_atoms(i) for i in range(min(n, ds.num_samples))]


def check_one(subset, data_dir, fmt, template, n_probe):
    """Resolve and probe one sub-dataset. Returns (path, fmt, problems, notes)."""
    problems = []
    notes = []

    try:
        path, fmt = resolve_input(data_dir, subset, fmt, template)
    except Exception as exc:
        return None, fmt, [str(exc)], notes

    try:
        frames = (first_frames_extxyz(path, n_probe) if fmt == "extxyz"
                  else first_frames_asedb(path, n_probe))
    except Exception as exc:
        return path, fmt, [f"could not parse {path}: {type(exc).__name__}: {exc}"], notes

    if not frames:
        return path, fmt, [f"{path} contains no frames"], notes

    a = frames[0]

    if "sid" not in a.info:
        problems.append(
            f"frame 0 has no info['sid'] (keys: {sorted(a.info)[:8]}). mat_id "
            f"comes from it, and stage 05 matches the same sid= token.")

    if not a.pbc.any():
        problems.append("frame 0 is not periodic; chemenv needs a cell.")
    elif abs(a.get_volume()) < 1e-6:
        problems.append("frame 0 has a degenerate cell (zero volume).")

    sids = [f.info.get("sid") for f in frames if "sid" in f.info]
    if sids and len(set(sids)) == 1 and len(sids) > 1:
        notes.append(f"the first {len(sids)} frames all share sid={sids[0]!r} — "
                     f"expected if a material has several configurations, "
                     f"suspicious otherwise")

    if not problems:
        notes.insert(0, f"{len(frames)} frames probed, "
                        f"{len(a)} atoms in frame 0, sid={a.info.get('sid')!r}")
    return path, fmt, problems, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ========================================================================
    # >>> USER CONFIGURATION -- pass these on the command line <<<
    #
    #   --data-dir   REQUIRED. The directory that HOLDS the sub-datasets named
    #                in subsets.txt (one <subset>/ directory or <subset>.xyz
    #                file per sub-dataset inside it) -- the same value you will
    #                give stage 01 as $OMAT_DATA_DIR.
    #   --format / --file-template  mirror $OMAT_INPUT_FORMAT / $OMAT_FILE_TEMPLATE.
    #   --subsets-file  the list of sub-datasets to check; defaults to the
    #                subsets.txt next to this script.
    # ========================================================================
    ap.add_argument("--data-dir", required=True,
                    help="directory HOLDING the sub-datasets named in "
                         "subsets.txt (same value as $OMAT_DATA_DIR for "
                         "stage 01), e.g. /data/omat24/train_extracted")
    ap.add_argument("--format", choices=["auto", "extxyz", "ase-db"], default="auto",
                    help="'auto' (default) uses whichever form is present per "
                         "sub-dataset, exactly as tokenize_omat24.py does")
    ap.add_argument("--file-template", default=None,
                    help="extxyz filename pattern, e.g. "
                         "'all_structures_{subset}.xyz'. Default: try "
                         "<subset>.xyz / .extxyz / .xyz.gz / .extxyz.gz")
    ap.add_argument("--subsets-file", default=DEFAULT_SUBSETS_FILE)
    ap.add_argument("--n-probe", type=int, default=3,
                    help="how many leading frames to parse per sub-dataset")
    args = ap.parse_args()

    subsets = load_subsets(args.subsets_file)
    print(f"Checking {len(subsets)} sub-datasets from {args.subsets_file}")
    print(f"  data-dir: {args.data_dir}   format: {args.format}\n")

    failed = []
    for name in subsets:
        path, fmt, problems, notes = check_one(
            name, args.data_dir, args.format, args.file_template, args.n_probe)
        status = "FAIL" if problems else "ok  "
        print(f"[{status}] {name}")
        if path:
            print(f"         [{fmt}] {path}")
        for n in notes:
            print(f"         note: {n}")
        for pr in problems:
            print(f"         ERROR: {pr}")
        if problems:
            failed.append(name)
        print()

    if failed:
        print(f"{len(failed)} of {len(subsets)} sub-datasets FAILED: "
              f"{', '.join(failed)}")
        return 1
    print(f"All {len(subsets)} sub-datasets look usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
