#!/usr/bin/env python3
"""
DIRECT step 6: write the selected structures into one training extxyz.

The DIRECT counterpart of stage 05's generate_train_set.py. Instead of the
sid= token, a frame is identified by its position: each selected id is
"<chunk basename>:<j>", frame j of that chunk, as assigned by
featurize_m3gnet.py. Chunks are streamed frame by frame and selected frames
are copied verbatim, so the output frames are byte-identical to OMat24.

USAGE
  python generate_train_set_direct.py \\
      --selected-ids selected_460k.ids.json \\
      --chunks-dir   chunks \\
      --output       subset_DIRECT_train.xyz
"""

import argparse
import json
import os
import time
from collections import defaultdict


def load_selected(path):
    """Return {chunk basename: set of frame indices} and the number of ids."""
    with open(path) as fh:
        ids = json.load(fh)
    per_chunk = defaultdict(set)
    for ident in ids:
        chunk, j = ident.rsplit(":", 1)
        per_chunk[chunk].add(int(j))
    return per_chunk, len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected-ids", required=True, help="<OUT>.ids.json from direct_sampling.py.")
    ap.add_argument("--chunks-dir", required=True,
                    help="the chunk directory featurize_m3gnet.py read.")
    ap.add_argument("--output", required=True, help="output xyz path.")
    ap.add_argument("--report-every", type=int, default=500_000,
                    help="print progress every N frames scanned.")
    args = ap.parse_args()

    t0 = time.time()

    per_chunk, n_requested = load_selected(args.selected_ids)
    print(f"Requested {n_requested:,} frames across {len(per_chunk):,} chunks "
          f"from {args.selected_ids}", flush=True)

    chunk_files = sorted(f for f in os.listdir(args.chunks_dir) if f.endswith(".xyz"))
    if not chunk_files:
        raise SystemExit(f"no .xyz chunk files under {args.chunks_dir}")
    present = set(chunk_files)
    absent = [c for c in per_chunk if c not in present]
    if absent:
        n_absent_frames = sum(len(per_chunk[c]) for c in absent)
        print(f"WARNING: {len(absent):,} referenced chunk file(s) "
              f"({n_absent_frames:,} frames) are not in {args.chunks_dir}; "
              f"examples: {absent[:3]}", flush=True)
    print(f"Found {len(chunk_files):,} chunk file(s).", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    n_scanned = 0
    n_written = 0
    remaining = {c: set(js) for c, js in per_chunk.items()}

    with open(args.output, "w") as out:
        for cf in chunk_files:
            want = per_chunk.get(cf)
            if not want:
                continue
            rem = remaining[cf]
            path = os.path.join(args.chunks_dir, cf)
            print(f"  Reading {cf} ({len(want):,} selected) ...", flush=True)
            with open(path) as f:
                j = 0
                while True:
                    line1 = f.readline()
                    if not line1:
                        break
                    if not line1.strip():
                        continue
                    n_atoms = int(line1.strip())
                    comment = f.readline()
                    atom_lines = [f.readline() for _ in range(n_atoms)]
                    n_scanned += 1
                    if args.report_every and n_scanned % args.report_every == 0:
                        print(f"    {n_scanned:,} frames scanned, "
                              f"{n_written:,} written ...", flush=True)

                    if j in want:
                        out.write(line1 + comment + "".join(atom_lines))
                        n_written += 1
                        rem.discard(j)
                    j += 1

    n_missing = sum(len(s) for s in remaining.values())
    print(f"\nDone in {(time.time()-t0)/60:.2f} min.")
    print(f"  frames scanned : {n_scanned:,}")
    print(f"  frames written : {n_written:,}  -> {args.output}")
    print(f"  requested      : {n_requested:,}")
    if n_missing:
        examples = [f"{c}:{j}" for c, s in remaining.items() if s for j in list(s)[:1]][:5]
        print(f"  WARNING: {n_missing:,} selected frames were NOT written "
              f"(absent chunk or index past end). Examples: {examples}")
    else:
        print(f"  all requested frames written (100%).")


if __name__ == "__main__":
    main()
