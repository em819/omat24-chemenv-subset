# Stage 05 — Extract the selected frames

**Env A** · `cpu` · 4 cores · 12 h

Turns the list of selected material ids into a training extxyz file.

## Files

`generate_train_set.py` · `submit_generate_train_set.sh`

## How it works

Streams the OMat24 extxyz files line by line, matches the `sid=` token in each
frame's comment line against the id list, and copies matching frames through
**verbatim** — the geometry is never parsed or rewritten, so the output is
byte-faithful to OMat24.

## Run

```bash
export SUBSET_IDS=/path/to/best_subset_....txt     # stage 04 output
export OMAT_XYZ_DIR=/path/to/omat24_extxyz         # the training split as extxyz
export TRAIN_XYZ=subset_train.xyz                  # -> stage 06's train_file
sbatch --account=<your allocation> submit_generate_train_set.sh
```

All three are documented in the `>>> USER CONFIGURATION <<<` block at the top of
`submit_generate_train_set.sh`; the first two are required. The Python script
itself is pure argparse (`--help`).

`--glob-pattern` defaults to `**/*xyz`, matched recursively, so it finds both
`<xyz-dir>/<subset>.xyz` (flat — what stage 01's auto-detection expects) and
`<xyz-dir>/<subset>/*.xyz` (nested — the as-run layout). Narrow it if your
directory contains xyz files that are not OMat24 source data; the script prints
how many files it matched and how many frames it scanned, so a wrong glob shows
up immediately in the counts.

## Input

- `SUBSET_IDS` — `best_subset_*.txt` from stage 04 (or the published id list,
  which lets you skip stages 01–04 entirely)
- `OMAT_XYZ_DIR` — the OMat24 training split **in extxyz form**

> This stage streams raw text and never parses a frame, which is what makes its
> output byte-faithful to OMat24. It therefore cannot read an ASE database and
> cannot read a gzipped file. Converting the OMat24 ASE databases to extxyz is
> outside the scope of this repository — see [`../DATA.md`](../DATA.md).

## Boundary check

64,648,643 frames scanned · **795,721 written** · 795,721/795,721 ids matched
(100%).

The scanned count equals the material count from stage 01 exactly. That equality
is the proof that stage 01's append-if-exists HDF5 write did not double-count
and that nothing was lost along the way — check it.
