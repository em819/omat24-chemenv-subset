# Stage 00 — Convert OMat24 to extxyz

**Env A** · `cpu` · 5 nodes × 128 MPI ranks per sub-dataset

Turns the public OMat24 ASE databases into the extxyz files that stage 05 and
the baselines in `04_subset_selection/` read, and drops every structure that
contains a rare earth or an actinide (outside the model's z-table).

## Files

`convert_omat24.py` · `submit_convert.sh`

## Run

Download the OMat24 training split from
[`fairchem/OMAT24`](https://huggingface.co/datasets/fairchem/OMAT24) and unpack
it, so that each sub-dataset of `01_token_construction/subsets.txt` is a
directory of `.aselmdb` files. Then:

```bash
export OMAT_DB_DIR=/path/to/omat24/train      # holds aimd-from-PBE-1000-npt/, ...
export OMAT_XYZ_DIR=/path/to/omat24_xyz/train
sbatch --account=<acct> submit_convert.sh     # array 0-10, one task per sub-dataset
```

Output: `$OMAT_XYZ_DIR/<subset>/all_structures_<subset>.xyz`. Point stage 05's
and the baselines' `OMAT_XYZ_DIR` at the same root.

The validation split is converted the same way (its own `OMAT_DB_DIR` and
`OMAT_XYZ_DIR`); stage 06 then takes the 11 files joined into one, e.g.
`cat $OMAT_XYZ_DIR/*/all_structures_*.xyz > omat24_validation_no_rare_earths.xyz`.

## What a frame carries

`ase.io.write(format="extxyz", write_info=True, write_results=True)` of the
`Atoms` that fairchem's `AseDBDataset` returns: `sid=` and the other OMat24
metadata, `energy`, `stress`, per-atom forces, and per-atom `initial_magmoms`
where the database has them, plus `multiplicity` = Σ |round(initial magmom)|.
The four `aimd-*` sub-datasets and `rattled-relax` carry essentially no
magnetic moments in OMat24, so most of their frames have no `initial_magmoms`
column.

Rank *r* of *N* converts the contiguous index block [r·n/N, (r+1)·n/N) and rank
0 joins the rank files in rank order, so the frames follow the database order
for any number of ranks.

## Relation to the paper's files

The paper's extxyz files were written by the as-run script with a dynamic MPI
task queue, so their frame order depends on that run's scheduling. The frames
themselves are the same: on an OMat24 shard of `aimd-from-PBE-1000-npt`
(46,636 structures, 2,792 with rare earths) this script and the as-run script
write the same 43,844 frames byte for byte, and the 4,592 of them that are in
the paper's subsets match the paper's files byte for byte. Only the order
differs, which matters for DIRECT (see `04_subset_selection/DIRECT/README.md`).

## Stage 01 reads the databases

Stage 01 tokenised the ASE databases directly in the paper. It can read these
extxyz files too (`OMAT_INPUT_FORMAT=extxyz`,
`OMAT_FILE_TEMPLATE='{subset}/all_structures_{subset}.xyz'`), but extxyz stores
positions with 8 decimals, which moves the chemenv geometry measures in the
last digits and can flip a borderline site's coordination label (1 of 1,500
sites in a 300-structure test). Use the databases for stage 01 to reproduce
the published tokens.
