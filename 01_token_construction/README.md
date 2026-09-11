# Stage 01 — Chemenv tokenisation of OMat24

**Env A** · `cpu` · 20 nodes × 128 ranks (2560 MPI) · 2 days **per sub-dataset**

The most expensive stage in the pipeline: ~440 node-days over all 11 OMat24
sub-datasets. If you only want the selected subset, skip to stage 05 and use the
published id list instead.

## Files

| File | Role |
|---|---|
| `check_input.py` | **run this first** — preflight validation of your input files |
| `input_backend.py` | locates each sub-dataset and decides ASE DB vs extxyz; shared by the tokeniser and the checker so they cannot disagree |
| `subsets.txt` | the 11 sub-dataset names, in order; read by the Python *and* the submit script |
| `tokenize_omat24.py` | per-site Chemenv tokenisation; writes one HDF5 per MPI rank |
| `submit_tokenize.sh` | SLURM job array `0-10`, one element per sub-dataset |
| `concat_to_parquet.py` | merges the rank shards into one parquet per sub-dataset |
| `submit_concat.sh` | loops over `subsets.txt`, 11 concurrent single-rank tasks |

## Step 0 — check your input

```bash
python check_input.py --data-dir /path/to/omat24
# non-default filenames:
python check_input.py --data-dir /path/to/omat24 \
                      --file-template 'all_structures_{subset}.xyz'
```

Needs only ASE and numpy; runs in seconds. It verifies for each sub-dataset that
the file parses and that frame 0 carries the three things the tokeniser reads:

| | Used for | If missing |
|---|---|---|
| a periodic cell | the geometry chemenv analyses | chemenv errors out |
| `info['sid']` | `mat_id`, and the key stage 05 matches on | `KeyError` on every frame |
| per-atom `initial_magmoms` | the `<spin>` field of **every token** | **silent corruption** — ASE returns zeros, so you get a complete, plausible, wrong token set |

`tokenize_omat24.py` repeats the same check on frame 0 and refuses to start if it
fails. `OMAT_ALLOW_MISSING_MAGMOMS=1` overrides it deliberately.

## Configure

| Variable | Default | Meaning |
|---|---|---|
| `OMAT_DATA_DIR` | *(the author's path)* | directory holding the sub-datasets |
| `OMAT_INPUT_FORMAT` | `auto` | `auto`, `ase-db` (fairchem `AseDBDataset`) or `extxyz` |
| `OMAT_FILE_TEMPLATE` | *(unset)* | extxyz filename pattern, e.g. `all_structures_{subset}.xyz` |
| `OMAT_DB_INDEX` | `0` | which line of `subsets.txt`; set from `SLURM_ARRAY_TASK_ID` |
| `OMAT_ALLOW_MISSING_MAGMOMS` | `0` | accept spin-0 tokens |

### Input format

Auto-detected **per sub-dataset**, so a directory holding some sub-datasets as
ASE databases and others as extxyz works without configuration:

| What is at `$OMAT_DATA_DIR/<subset>` | Backend |
|---|---|
| a directory | `ase-db` — fairchem `AseDBDataset`, the as-run path |
| `<subset>.xyz`, `.extxyz`, `.xyz.gz` or `.extxyz.gz` | `extxyz` |
| whatever `OMAT_FILE_TEMPLATE` names | `extxyz` |
| **both** a directory and a file | **refuses** — they may hold different data, so set `OMAT_INPUT_FORMAT` explicitly |
| neither | error naming everything it tried |

Set `OMAT_INPUT_FORMAT=ase-db` or `=extxyz` to force one and get a hard error
rather than a silent fallback if it is absent.

The sub-dataset label written into the HDF5 filenames and the `omat_subset`
column is the name from `subsets.txt`, not anything derived from the filename —
so `OMAT_FILE_TEMPLATE` cannot leak a decorated name into it, and the label
always matches the working-directory name stage 01b derives its own label from.

Sub-dataset names live in `subsets.txt` — edit there, not in the source.

In `submit_concat.sh`, `EXPECTED_RANKS` (default 2560) is a completeness check
only: it warns if fewer shards are found than ranks that should have written
one. The two 3000-atom AIMD sets were re-run at 10240 ranks in the published
run, so pass `EXPECTED_RANKS=10240` for those if you reproduce that.

## Run

```bash
export OMAT_DATA_DIR=/path/to/omat24
sbatch submit_tokenize.sh      # array 0-10; format auto-detected
# ... once all array elements have finished:
sbatch submit_concat.sh
```

Without SLURM:

```bash
export OMAT_DB_INDEX=6                       # rattled-300
mkdir -p rattled-300 && cd rattled-300
mpirun -n 256 python ../tokenize_omat24.py
cd .. && python concat_to_parquet.py rattled-300 256
```

## Directory layout — mandatory

Stage 01a writes `chemenv_<subset>_rank<N>.hdf5` into the **current directory**,
and stage 01b derives the sub-dataset label from the **name of the directory** it
is handed. So each sub-dataset must be tokenised in its own directory, named
exactly as in `subsets.txt`:

```
work/
├── aimd-from-PBE-1000-npt/     ← array index 0 runs here
│   └── chemenv_aimd-from-PBE-1000-npt_rank*.hdf5
├── …one per sub-dataset…
└── parquet_file_collection/    ← stage 01b writes here
    └── total_chemenv_<subset>.parquet
```

`submit_tokenize.sh` creates and enters the right directory automatically. Run
two sub-datasets in one directory and stage 01b will glob both sets of shards
and label the result after the wrong sub-dataset.

The HDF5 write *appends* if the file exists, so a leftover shard from an aborted
attempt is double-counted. Start each attempt from a clean directory.

## Output

`chemenv_<subset>_rank<N>.hdf5` → `parquet_file_collection/total_chemenv_<subset>.parquet`,
which is exactly where stage 02 looks.

## Notes

- **Token grammar.** `<element>_<spin>_<neighbours>_<polyhedron>_<f5>_<dmin>_<a_bin>`.
  The field count is not fixed — 11 Chemenv symbols contain underscores — so
  trailing fields must be indexed from the END of the token.
- **`structure_refinement=STRUCTURE_REFINEMENT_NONE` is essential.** Refinement
  would reorder sites and break index alignment with the input structure.
- The extxyz reader is the byte-offset index from stage 10.2, including
  `parallel=False` — ASE defaults to `parallel=True`, which under a >1-rank MPI
  world makes only rank 0 parse and broadcasts its result, so every rank would
  silently get rank 0's structure.

## Boundary check

194,928,651 tokens · 1,180,921,883 sites · 64,648,643 materials.
