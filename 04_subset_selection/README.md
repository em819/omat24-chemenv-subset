# Stage 04 — Subset selection

**Env A** · `cpu` · 1 node, 128 cores, **400 GB RAM** · 2 days

The core of the method. Builds demand classes from an iterative coarsening
ladder over the token grammar, then runs a lazy-greedy (CELF) set cover over
materials until every class is satisfied.

## Files

`select_subset.py` (~4,800 lines) · `submit_select_subset.sh`

## Configure

Paths and resources come from the environment; the selection knobs live in
`main()` and each has an override. The script raises `FileNotFoundError` if
either input database is missing.

| Variable | Default | Meaning |
|---|---|---|
| `V10_TOKEN_VEC_DB` | `material_vectors.duckdb` | stage 03 output |
| `V10_ENV_FREQ_DB` | `freq_chemenv.duckdb` | stage 02 output |
| `V10_THREADS` | 128 | DuckDB threads |
| `V10_MEMORY` | `400GB` | what DuckDB may take — see the memory note below |
| `V10_T`, `V10_DESIGN`, `V10_SUPPORT`, `V10_NCOV`, `V11_COST_ALPHA` | 100, `a`, `materials`, 1, 1 | the published settings |

| Knob | Value | Meaning |
|---|---|---|
| `coarsening_T` | 100 | support threshold for a demand class |
| `coarsening_design` | `'a'` | strict partition (vs `'d'`) |
| `support_metric` | `'materials'` | count materials, not sites |
| `n_coverage` | 1 | each class needs one covering material |
| `uniformity_weight` | 0.0 | uniformity term off |
| `cost_alpha` | 1.0 | `score = (help + uw·uniformity) / site_count**alpha` |
| `enable_floors` | False | no pre-selection floors |
| `enable_floor_topup` | True | Approach-F floors enforced after selection |
| `enable_dist_floor`, `enable_asym_floor` | True | distance and asymmetry floors |
| `enable_full_token_rung` | True | full-token rung at the bottom of the ladder |
| `enable_dedup` | False | |
| `enable_forced_selection` | True | |
| `enable_redundancy_prune` | True | post-greedy prune |
| `restart_interval` | 50000 | |
| `duck_threads` / `duck_memory` | 128 / `400GB` | |

The ladder is a lexicographic product of identity levels and a modifier chain,
**modifier outer**:

```python
_MODIFIER_CHAIN = [("dA", "||'|'||dmin||'|'||a_bin"), ("A", "||'|'||a_bin"), ("", "")]
```

That nesting order is what version 14 changed, and it is what turns wall-atom
depletion into enrichment.

## Run

```bash
sbatch submit_select_subset.sh     # sets NUMBA_NUM_THREADS=128
```

## Output

- `best_subset_*.txt` — one selected material id per line (795,721 lines)
- `env_ids_found_distribution.txt` — per-token atom counts in the subset

## Notes

> **Run this in its own empty working directory.** `restart.npz` and
> `env_ids_found_distribution.txt` are fixed filenames. A leftover `restart.npz`
> makes a fresh run silently resume the previous one instead of starting over.

**The memory requirement is not really configurable.** `V10_MEMORY` controls what
DuckDB is *allowed* to take, but the Python side alone peaks near 165 GiB holding
`mat_vecs_dict` and the CSR matrix (measured system-wide peak on the T=30 run:
188 GiB), and that allocation has no knob. 400 GB was chosen over 500 GB
precisely so the two allocators cannot both overcommit on a 503 GiB node. On a
smaller machine you can only run a smaller corpus.

## Boundary check

3,443,621 demand classes at T=100 · 3 forced materials · greedy 837,273 →
redundancy prune −5.11% → 794,511 → floor top-up +1,210 → **795,721 materials /
6,566,253 atoms**. All four Approach-F floors 100% present after top-up; all
demand classes satisfied.

`env_ids_found_distribution.txt` holds the per-token atom counts of the selected
subset. Compare it against the corpus-wide counts in `chemenv_frequency_table`
to see which environments the selection enriched.
