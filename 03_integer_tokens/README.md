# Stage 03 — Integer token ids and material vectors

**Env A** · `cpu` · 128 cores · 2 days

Maps the string tokens to integer ids and builds, for each material, the vector
of token ids it contains with their counts. This is the representation stage 04
runs set cover over.

## Files

`build_integer_tokens.py` · `submit_integer_tokens.sh`

## Configure

All via the environment; the source needs no edit. The script raises
`FileNotFoundError` if the input database is absent rather than letting DuckDB
create an empty one.

| Variable | Default | Meaning |
|---|---|---|
| `CHEMENV_FREQ_DB` | `freq_chemenv.duckdb` | stage 02 output |
| `TOKEN_VEC_DB` | `material_vectors.duckdb` | output; stage 04 reads it |
| `DUCK_THREADS` | 128 (or `$SLURM_CPUS_PER_TASK`) | |
| `DUCK_MEMORY` | `400GB` | the one to lower on a smaller machine |
| `DUCK_TEMP_DIR` | `./duckdb_temp` | spill directory; can reach several hundred GB on the full corpus |

## Run

```bash
sbatch submit_integer_tokens.sh
```

## Output

| Table / file | Contents |
|---|---|
| `environment_mapping` | token string → integer id |
| `env_counts` | one row per (material, token string): `mat_id`, `env_id`, `count` = occurrences of that token in that material |
| 1000 batch parquets | sharded by `hash(mat_id) % 1000` |
| `material_vectors` | material → (token ids, counts), with `mat_index = ROW_NUMBER() OVER (ORDER BY mat_id) - 1` |

## Notes

Counts are **not** normalised. The normalisation block is deliberately commented
out — selection operates on raw per-material token counts, and normalising here
would silently change the cost model in stage 04.

`mat_index` is a dense 0-based index over materials sorted by `mat_id`; stage 04
uses it to address its flat CSR-shaped buffers, so the ordering must not change
between the two stages.
