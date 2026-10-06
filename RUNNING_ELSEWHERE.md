# Running this pipeline somewhere else

Target scenario: **you have the 11 OMat24 training sub-datasets as extxyz files
and the four environments installed. You are not on MeluXina.**

This document says what works, what you must provide, and what is genuinely out
of reach.

---

## Short answer

Stages **01 → 08** (tokenise → select → train → export) are runnable from extxyz
alone. Stage 01 reads either the OMat24 ASE databases or extxyz files —
auto-detected per sub-dataset, so a mixed directory works too — and every
absolute path in stages 01–04 is an environment variable.

Two things are **not** reachable from the stated starting point, and no change
to this repository can fix them:

- **Stage 06 needs a validation and a test set**, which are not among the 11
  training sub-datasets. Bring your own, or train on the subset alone.
- **Stage 09's four benchmarks each need their own reference dataset** (MatPES,
  MatCalc-Bench, MLIP-Arena, LiTraj nebDFT2k), fetched by their own tooling.

Everything else is a matter of setting variables and having the hardware.

---

## What your extxyz files must contain

Per frame, the tokeniser reads exactly three things:

| | Where | Used for | If missing |
|---|---|---|---|
| ASE `Atoms` (positions, species, **periodic cell**) | the frame | the geometry chemenv analyses | chemenv errors out |
| `info['sid']` | the comment line, as `sid=<id>` | `mat_id`; also the key stage 05 matches on | `KeyError` on every frame |
| per-atom `initial_magmoms` (where available) | a `Properties=` column | the `<spin>` field of every token | spin 0 for those atoms, as in the paper: the `aimd-*` sub-datasets and `rattled-relax` carry no magmoms in OMat24 |

Check before committing any compute:

```bash
cd 01_token_construction
python check_input.py --data-dir /path/to/omat24
# if your files are named differently:
python check_input.py --data-dir /path/to/omat24 \
                      --file-template 'all_structures_{subset}.xyz'
```

It needs only ASE and numpy, runs in seconds, and reports per sub-dataset
whether the file parses and carries `sid` and a periodic cell. It
resolves each sub-dataset through the same module the tokeniser uses
(`input_backend.py`) and prints which backend it picked, so it doubles as a check
that auto-detection sees what you expect. Stage 01 repeats the frame-0 `sid`
check itself and refuses to start if it fails.

Sub-dataset names and order live in `01_token_construction/subsets.txt`. Both
the Python and the SLURM script read that one file — edit it there if your file
set differs.

---

## Hardware you actually need

Not adjustable by configuration:

| Stage | Hard requirement | Why |
|---|---|---|
| 01 | a working MPI, and realistically ~2,500 cores | `MPICommExecutor`; ~440 node-days total on 128-core nodes |
| 03 | one machine, large RAM | single DuckDB process over ~1.2 × 10⁹ site rows |
| 04 | **one machine with ~400 GB RAM** | the Python side alone peaks near 165 GiB holding the CSR matrix and material-vector dict; DuckDB wants the rest |
| 07 | multi-GPU | 8 × A100 × 2 days for the published model |

Stage 04 is the real wall. `V10_MEMORY` controls what DuckDB is *allowed* to
take, but the ~165 GiB of Python-side allocation is not configurable — that is
the floor. On a smaller machine you can only run a smaller corpus.

Stages 02 and 03 spill to `DUCK_TEMP_DIR` rather than failing when
`DUCK_MEMORY` is low, so they degrade gracefully at the cost of disk and time.
Give that directory several hundred GB on the full corpus.

## SLURM is not required

Nothing in the Python depends on SLURM. The submit scripts are convenience
wrappers; every stage can be launched directly:

```bash
# stage 01, one sub-dataset, under plain MPI
export OMAT_DATA_DIR=/path/to/omat24 OMAT_DB_INDEX=6
mkdir -p rattled-300 && cd rattled-300
mpirun -n 256 python ../tokenize_omat24.py

# stages 02, 03, 04 are plain single-process jobs
python build_frequency_db.py
python build_integer_tokens.py
python select_subset.py
```

What the submit scripts do carry that you must not lose: the **one working
directory per sub-dataset** rule in stage 01 (below), the thread-limiting
exports (`OMP_NUM_THREADS=1` etc., so DuckDB and numba do not oversubscribe),
and `NUMBA_NUM_THREADS` in stage 04.

---

## Directory layout stages 01–04 assume

This is the one piece of implicit structure in the pipeline. It was implicit in
the original too — eleven hand-made directories — and it is now enforced by
`submit_tokenize.sh`, but if you launch by hand you must reproduce it.

```
work/
├── aimd-from-PBE-1000-npt/        ← stage 01a runs HERE for index 0
│   └── chemenv_aimd-from-PBE-1000-npt_rank*.hdf5
├── …one directory per sub-dataset, named exactly as in subsets.txt…
├── rattled-relax/
├── parquet_file_collection/       ← stage 01b writes here
│   └── total_chemenv_<subset>.parquet
└── selection/                     ← stages 02–04 run here
    ├── freq_chemenv.duckdb
    ├── material_vectors.duckdb
    └── best_subset_*.txt
```

**Why the per-sub-dataset directory is mandatory:** stage 01a writes its shards
into the *current* directory, and stage 01b derives the sub-dataset label from
the *directory name* it is handed. Run two sub-datasets in one directory and
stage 01b will glob both sets of shards and label the result after the wrong
sub-dataset.

Stage 02's default parquet glob is `../parquet_file_collection/…`, which is why
stages 02–04 run one level down. Override `CHEMENV_PARQUET_GLOB` if you prefer
another layout.

---

## Configuration reference

Everything site-specific is an environment variable. Nothing in stages 01–04
needs the source edited.

### Stage 01

| Variable | Default | Meaning |
|---|---|---|
| `OMAT_DATA_DIR` | *(required — no default)* | directory holding the sub-datasets |
| `OMAT_INPUT_FORMAT` | `auto` | `auto`, `ase-db` or `extxyz` |
| `OMAT_FILE_TEMPLATE` | *(unset)* | extxyz filename pattern, e.g. `all_structures_{subset}.xyz` |
| `OMAT_DB_INDEX` | `0` | which line of `subsets.txt` |

Under `auto`, each sub-dataset is resolved on its own: a **directory** at
`$OMAT_DATA_DIR/<subset>` means ASE DB, a **file** at `<subset>.xyz` (or
`.extxyz`, `.xyz.gz`, `.extxyz.gz`, or whatever `OMAT_FILE_TEMPLATE` names)
means extxyz. Half-and-half is fine. If a sub-dataset is present in **both**
forms the run refuses to start rather than guess — they may hold different data.
Forcing `ase-db` or `extxyz` turns a missing form into a hard error instead of a
fallback.

### Stages 02 / 03

| Variable | Default | Meaning |
|---|---|---|
| `CHEMENV_PARQUET_GLOB` | `../parquet_file_collection/total_chemenv*.parquet` | stage 01b output (stage 02) |
| `CHEMENV_FREQ_DB` | `freq_chemenv.duckdb` | stage 02 output / stage 03 input |
| `TOKEN_VEC_DB` | `material_vectors.duckdb` | stage 03 output |
| `DUCK_THREADS` / `DUCK_MEMORY` / `DUCK_TEMP_DIR` | 128 / `400GB` / `./duckdb_temp` | DuckDB resources |

### Stage 04

| Variable | Default | Meaning |
|---|---|---|
| `V10_TOKEN_VEC_DB` | `material_vectors.duckdb` | stage 03 output |
| `V10_ENV_FREQ_DB` | `freq_chemenv.duckdb` | stage 02 output |
| `V10_THREADS` / `V10_MEMORY` | 128 / `400GB` | DuckDB resources |
| `V10_T`, `V10_DESIGN`, `V10_SUPPORT`, `V10_NCOV`, `V11_COST_ALPHA` | 100, `a`, `materials`, 1, 1 | the published selection settings |

### Stage 05

| Variable | Default | Meaning |
|---|---|---|
| `SUBSET_IDS` | *(required)* | the `best_subset_*.txt` written by stage 04 |
| `OMAT_XYZ_DIR` | *(required)* | root directory of the OMat24 training split **as extxyz** |
| `TRAIN_XYZ` | `subset_train.xyz` | output; becomes stage 06's `train_file` |
| `GLOB_PATTERN` | `**/*xyz` | narrow it if the directory holds other xyz files |

### Stages 06 – 08

| Variable | Default | Meaning |
|---|---|---|
| `MACE_VENV` | *(required)* | the env B virtualenv directory |
| `CONFIG` | the stage's own YAML | `preprocess.yaml` / `train.yaml` / `compile.yaml` |
| `SHARD_DIR` | `./preprocess_shards_$SLURM_JOB_ID` | stage 06 scratch |
| `READ_PROCS` / `STATS_PROCS` / `TRAIN_SHARDS` | 128 | stage 06 parallelism |

The three dataset paths in `preprocess.yaml` ship as `CHANGE_ME` placeholders;
`preprocess_parallel.py` refuses to start until you edit them, naming the key.

### Stage 09

| Variable | Default | Meaning |
|---|---|---|
| `MATCALC_VENV` / `ARENA_VENV` / `MACE_VENV` | *(required)* | env C / env D / env B |
| `MODEL` | *(required)* | `_compiled.model` for matcalc and matpes, `_run-1.model` for stability and nebdft2k |
| `MODEL_NAME` | a descriptive default | goes into every output filename; keep it stable between run and aggregate |
| `OUTDIR` | `$SLURM_SUBMIT_DIR/results…` | outputs |
| `TESTSET` | *(required, matpes)* | the MatPES extxyz |
| `NEB_CACHE_DIR` | *(required, nebdft2k)* | the prefetched nebDFT2k dataset |
| `MATCALC_CACHE` / `HF_CACHE` | `~/.cache/…` | presence checks for the staged reference data |
| `MATPES_RUNS` | *(required, aggregate)* | directory holding the per-model `*_paper` directories |
| `RUN_DIRS` / `LABELS` / `CEILING_DIR` | the paper's seven models | edit in the script; the two arrays must be the same length |

### Stage 10

| Variable | Default | Meaning |
|---|---|---|
| `REF_XYZ` | *(required, 10.1)* | the reference extxyz with DFT energies/forces |
| `MODEL` | *(required, 10.1)* | the model to evaluate |
| `OUTDIR` | `..` | must be the **parent** of the directory 10.2 runs in |
| `CHEMENV_PRED_XYZ` | `../all_structures_ref_and_predicted.xyz` | 10.2 input |
| `CHEMENV_OUT_XYZ` | `all_structures_with_errors_and_v4chemenvs_…xyz` | 10.2 output; 10.3 must be pointed at the same name |
| `XYZ` | 10.2's default output | 10.3 input |
| `FEATURE_MODEL` | *(required, 10.3)* | the model features are extracted from — **the same one 10.1 used**, and never a `_compiled.model` |
| `OUT_DIR` | `$SLURM_SUBMIT_DIR/mace_features` | 10.3 output |
| `FEAT_DIR` | `../3_extract_features/mace_features` | 10.4 and 10.5 (neighbour composition) input |
| `REPORT_OUT` | `$SLURM_SUBMIT_DIR/report_card.parquet` | 10.4 output |
| `KNN_BACKEND` | `faiss-gpu` | kNN backend. 10.4: `auto`, `faiss`, `faiss-gpu` or `ckdtree` (`ckdtree` without a GPU); 10.5 neighbour composition: `faiss` (CPU) or `faiss-gpu` |
| `REPORT_CARD` | `../4_cluster_metrics/report_card.parquet` | 10.5 input (the 10.4 output) |
| `TOKEN_VEC_DB` | *(required, 10.5 train coverage)* | stage 03's `material_vectors.duckdb` |
| `COVERAGE_OUT` | `train_coverage.parquet` | 10.5 train-coverage output |
| `MIN_ATOMS` | `30` | 10.5 train coverage: only descriptors with at least this many atoms get a row. Keep it equal to `split_descriptors.py --min-atoms`; a larger value makes descriptors without a row count as absent from training |
| `OUT_JSON` | `$SLURM_SUBMIT_DIR/neighbour_composition.json` | 10.5 neighbour-composition output |

### Everywhere

| Variable | Default | Meaning |
|---|---|---|
| `CHEMENV_CONDA_ENV` | `active_learning_omat24_python310` | the conda env holding env A (stages 01–05, 10.2, 10.5 train coverage) |

`#SBATCH --account` cannot read a shell variable, so it stays `CHANGE_ME` in
every submit script — edit it, or override at submit time with
`sbatch --account=… --partition=…`.

---

## What you must supply beyond the 11 training sub-datasets

### A validation and a test set (stage 06)

`preprocess.yaml` names three inputs. The published run used the **official
OMat24 validation split** (657,756 frames) for validation and **MatPES PBE
2025.2** (323,096 frames, rare earths removed) as the test set. Neither comes
from the 11 training sub-datasets.

Options, in order of fidelity:

1. Obtain the OMat24 validation split and MatPES as extxyz and point
   `valid_file` / `test_file` at them. Reproduces the published setup.
2. Hold out part of the selected subset as validation and drop `test_file`.
   Trains a comparable model; the loss curve is no longer comparable to the
   paper's.

Either way, filter out rare-earth elements — the model's z-table is
{1…56} ∪ {72…83}, and `preprocess.yaml`'s `atomic_numbers` encodes exactly that.

### `E0s` in `preprocess.yaml`

The 89 isolated-atom reference energies must match the DFT setup of your data.
The shipped values are OMat24's PBE E0s. If your extxyz came from OMat24 they
are correct as they stand.

### Benchmark reference data (stage 09)

MatCalc-Bench, stability and nebDFT2k each fetch their own reference data, and
compute nodes are usually offline, so stage it on a login node first. `nebdft2k`
has an explicit `--prefetch` mode, and its submit script stops with that command
if the data is missing. The matcalc and stability submit scripts only warn when
their cache is missing and continue (the library then tries to download, which
fails on an offline node); set `BENCH_OFFLINE=1` to make them stop instead.
MatPES reads an extxyz you supply (`TESTSET`). See `09_benchmarks/README.md`.

---

## Known non-portable remnants

Small, deliberate, and documented rather than changed:

- **EasyBuild `module load` lines** are MeluXina-specific. Delete them if your
  MPI, HDF5 and CUDA are already on the path.
- **`#SBATCH --account=CHANGE_ME`** everywhere, because a SLURM directive cannot
  read a shell variable. Overridable at submit time.
- The conda environment name defaults to `active_learning_omat24_python310`;
  set `CHEMENV_CONDA_ENV` rather than editing the scripts.

No absolute path from the original cluster remains anywhere in the scripts. Each
one is a named variable in that script's `>>> USER CONFIGURATION <<<` block, and
an input that cannot be guessed has no default at all — the job stops at once,
naming the variable, rather than reading someone else's data.

---

## Suggested order of attack

1. `check_input.py` — minutes, and it tells you early whether stage 01 can read
   your files.
2. Stage 01 on **one** sub-dataset, with a small `--array` of one and a short
   walltime, to confirm the tokens look right before committing 440 node-days.
   Inspect a few `env_id` values in the resulting HDF5.
3. Stage 01b on that one sub-dataset, then stages 02–04 on it alone. The
   selection runs fine on a single sub-dataset and gives you a small,
   end-to-end-verified pipeline in hours rather than weeks.
4. Only then scale to all 11.
