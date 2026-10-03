# Stage 04 (baseline) — random subsets

**Env A** · `cpu` · 1 node, 128 cores · < 1 h (selection) + one pass over the
OMat24 extxyz

Uniformly random subsets of the OMat24 training split, one per seed. The paper
used four (`rdm1`…`rdm4`), which set the run-to-run noise floor of every
benchmark.

## Files

`select_random_subset.py` · `submit_random_subset.sh`

## How it works

1. **Pool:** materials from the `material_elements` table of stage 02's
   `freq_chemenv.duckdb` whose elements all lie in the 68-element z-table
   ({1…56} ∪ {72…83}) — all 64,648,643.
2. **Reserve:** walking the elements alphabetically, one material for each
   element not yet covered → **35 materials** covering all 68 elements. The
   reserved set does not depend on the seed.
3. **Fill:** `N − 35` materials by `USING SAMPLE reservoir(N ROWS) REPEATABLE(seed)`.
4. **Write:** stream the OMat24 extxyz (`<dir>/<subset>/*xyz`, sorted) and copy
   every selected frame verbatim into `train.xyz`.

## Run

```bash
export CHEMENV_FREQ_DB=/path/to/freq_chemenv.duckdb   # stage 02
export OMAT_XYZ_DIR=/path/to/omat24_extxyz            # as for stage 05
for s in <seed1> <seed2> <seed3> <seed4>; do
    RANDOM_SEED=$s sbatch --account=<acct> submit_random_subset.sh
done
```

Output per seed, in `random_seed<seed>/`: `random_subset_mat_ids.txt` and
`train.xyz` (→ stage 06 `train_file`).

## Boundary checks (from the paper's run)

`Pool size: 64,648,643` · `Reserved 35 materials covering 68 elements` ·
356,249 materials selected · ~6.5 M atoms per subset (6,496,886 – 6,513,135).

## Notes

- **The reservoir sample can return slightly fewer rows than requested.** On
  the full pool one published subset has 356,247 structures instead of 356,249,
  and a small local test reproduces the effect (2,996 of 2,997). This is DuckDB
  behaviour, carried over unchanged; the id list and `train.xyz` then hold
  fewer than `N_TARGET` structures.
- The sample depends on the DuckDB version and on the order of the pool table.
  To rebuild the published subsets exactly, use the deposited id lists with
  stage 05 instead.
- Only the `material_elements` table of stage 02 is read, so stages 01–02 are
  needed for this baseline.
