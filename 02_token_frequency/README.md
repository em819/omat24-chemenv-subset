# Stage 02 — Token frequency database

**Env A** · `cpu` · 1 core · 1 day

Loads the stage-01 parquet files into DuckDB and builds the tables the rest of
the pipeline queries.

## Files

`build_frequency_db.py` · `submit_frequency.sh`

## Configure

All via the environment; the source needs no edit.

| Variable | Default | Meaning |
|---|---|---|
| `CHEMENV_PARQUET_GLOB` | `../parquet_file_collection/total_chemenv*.parquet` | stage 01b output |
| `CHEMENV_FREQ_DB` | `freq_chemenv.duckdb` | output database |
| `DUCK_THREADS` | 128 (or `$SLURM_CPUS_PER_TASK`) | |
| `DUCK_MEMORY` | `400GB` | lower on a smaller machine |
| `DUCK_TEMP_DIR` | `./duckdb_temp` | spill directory — needs room |

The default glob assumes this job runs in a **sibling directory** of the one
stage 01b ran in. Override it if you prefer another layout.

When `DUCK_MEMORY` is too small for the corpus DuckDB spills to `DUCK_TEMP_DIR`
rather than failing, so this degrades gracefully at the cost of disk and time.

## Run

```bash
sbatch submit_frequency.sh
```

## Output

`freq_chemenv.duckdb`, containing:

| Table | Contents |
|---|---|
| `data` | every site, with its token and material id |
| `chemenv_frequency_table` | token → occurrence count (indexed on `env_id`) |
| `material_elements` | material id → element list |

## Notes

This database is read by **both** stage 03 and stage 04 — it is a shared input,
not a throwaway intermediate. Do not delete it after stage 03.

Single-core and a full day: the cost is DuckDB's parquet scan, not compute.
