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
sbatch submit_generate_train_set.sh
```

Arguments (see `--help`): the id list, the input extxyz files, the output path.

`--glob-pattern` defaults to `**/*xyz`, matched recursively, so it finds both
`<xyz-dir>/<subset>.xyz` (flat — what stage 01's auto-detection expects) and
`<xyz-dir>/<subset>/*.xyz` (nested — the as-run layout). Narrow it if your
directory contains xyz files that are not OMat24 source data; the script prints
how many files it matched and how many frames it scanned, so a wrong glob shows
up immediately in the counts.

## Input

- `best_subset_*.txt` from stage 04
- the OMat24 training split in extxyz form

> The extxyz files are produced from the OMat24 ASE databases by a conversion
> step that is outside the scope of this repository. Stage 05 assumes they
> already exist.

## Boundary check

64,648,643 frames scanned · **795,721 written** · 795,721/795,721 ids matched
(100%).

The scanned count equals the material count from stage 01 exactly. That equality
is the proof that stage 01's append-if-exists HDF5 write did not double-count
and that nothing was lost along the way — check it.
