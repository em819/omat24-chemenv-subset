# Data you need to run this pipeline

This repository contains **scripts only**. Everything they read comes from
somewhere else. This document lists each input, where it comes from, which stage
consumes it, and which variable you point at it.

Every path in every script is a named variable in a
`>>> USER CONFIGURATION <<<` block at the top of that script. Nothing has a
default pointing into anyone's filesystem: a required input that you have not
set stops the job immediately with a message naming the variable.

---

## At a glance

| # | Data | Public? | Needed by | Variable |
|---|---|---|---|---|
| 1 | OMat24 training split, 11 sub-datasets | yes | 01, 05 | `OMAT_DATA_DIR`, `OMAT_XYZ_DIR` |
| 2 | OMat24 validation split, rare earths removed | yes | 06, 10.1 | `valid_file`, `REF_XYZ` |
| 3 | MatPES PBE 2025.2, rare earths removed | yes | 06, 09/matpes | `test_file`, `TESTSET` |
| 4 | MatCalc-Bench reference data (4 files) | yes | 09/matcalc | `MATCALC_CACHE` |
| 5 | MLIP-Arena RM24 structure database | yes | 09/stability | `HF_CACHE` |
| 6 | LiTraj nebDFT2k (65 MB) | yes | 09/nebdft2k | `NEB_CACHE_DIR` |
| 7 | MACE-OMAT-0 medium checkpoint | yes | 10.1, 10.3, 09/matpes aggregate | `FEATURE_MODEL`, `MODEL` |
| 8 | Six comparison models (DIRECT + 4 replicas + OMAT-0) | OMAT-0 yes; the rest via `04_subset_selection/{DIRECT,random}` + 06–08 | 09/matpes aggregate only | `RUN_DIRS` |
| 9 | The published subset id list | archive | 05 (shortcut past 01–04) | `SUBSET_IDS` |

Items 1–7 and 9 are obtainable. Of item 8, MACE-OMAT-0 is public; the DIRECT
and random models are trained with stages 06–08 on the subsets built in
`04_subset_selection/DIRECT` and `04_subset_selection/random` (or on the
deposited id lists, which reproduce the published subsets exactly).

---

## 1. OMat24 training split — the one hard input

Source: [`fairchem/OMAT24`](https://huggingface.co/datasets/fairchem/OMAT24)
(Meta FAIR). The 11 sub-datasets and their canonical order are in
[`01_token_construction/subsets.txt`](01_token_construction/subsets.txt).

Scale, from the published run: **64,648,643 frames · 1,180,921,883 sites**.
As uncompressed extxyz that is on the order of 100–200 GB.

**You need it in two forms, and stage 05 is the strict one:**

| Stage | Accepts | Variable |
|---|---|---|
| 01 | ASE database directories **or** extxyz files, auto-detected per sub-dataset | `OMAT_DATA_DIR` |
| 05 | **extxyz only, uncompressed** | `OMAT_XYZ_DIR` |

Stage 05 streams raw text and matches the `sid=` token without ever parsing a
frame, which is what makes its output byte-identical to the source. That also
means it cannot read an ASE database, and it cannot read a `.gz`. If your copy
of OMat24 is in `.aselmdb` form, convert it with stage 00
(`00_convert_omat24/`), which keeps `sid` and `initial_magmoms` and drops the
rare-earth structures.

### What every frame must carry

A periodic cell and `sid` are required; magnetic moments are used where present:

| | Where | Used for | If missing |
|---|---|---|---|
| a periodic cell | the frame | the geometry chemenv analyses | chemenv errors out |
| `info['sid']` | `sid=<id>` in the comment line | `mat_id`, and the key stage 05 matches on | `KeyError` on every frame |
| per-atom `initial_magmoms` (where available) | a `Properties=` column | the `<spin>` field of every token | spin 0 for those atoms, as in the paper: the `aimd-*` sub-datasets and `rattled-relax` carry no magmoms in OMat24 |

Check before committing any compute — it takes seconds:

```bash
python 01_token_construction/check_input.py --data-dir /path/to/omat24
```

## 2. Validation set — OMat24 official validation split

**657,756 frames**, extxyz, rare earths removed. Public (the `val` split of the
same HuggingFace dataset); convert it with stage 00 like the training split
and join the 11 files into one (see `00_convert_omat24/README.md`).

Consumed by:
- stage 06, `valid_file` in `preprocess.yaml`
- stage 10.1, `REF_XYZ` — the reference set the whole latent-space analysis is
  sized by (the 657,756-structure / 12,019,558-atom boundary check)

Without it, stage 06 can still run: hold out part of the stage-05 training set
and drop `test_file`. The model stays comparable; the loss curve no longer is.

## 3. Test set — MatPES PBE 2025.2

**323,096 frames**, extxyz with `energy` / `forces` / `stress`, rare earths
removed. Public from MatPES; conversion and filtering are yours to do.

Consumed by stage 06 (`test_file`) and stage 09/matpes (`TESTSET`). The two must
be the same file — the MatPES benchmark's 323,096-structure boundary check is
exactly the stage-06 test count.

### Rare-earth filtering is not optional

The model's z-table is {1…56} ∪ {72…83}, encoded in `atomic_numbers` in both
`preprocess.yaml` and `train.yaml`. An out-of-table element is a hard failure.
Stage 01 drops rare-earth structures itself (`contains_rare_earths`); the
validation and test sets you supply must be filtered the same way.

### E0s

`preprocess.yaml` ships 89 isolated-atom reference energies. They are OMat24's
PBE values and are correct as they stand **if your training data is OMat24**.
Replace them if it is not.

## 4–6. Benchmark reference data

Each benchmark downloads its own. Compute nodes are often offline, so stage
these from a machine with internet first. When the data is missing, the
MatCalc-Bench and stability submit scripts warn and continue (the library then
tries to download), and `BENCH_OFFLINE=1` makes them stop instead; the nebDFT2k
submit script always stops, printing the `--prefetch` command.

| Benchmark | What | How to stage it |
|---|---|---|
| MatCalc-Bench | 4 files from HF `materialyze/matcalc-bench`: `wbm-random-pbe52-equilibrium-2025.1.json.gz`, `mp-binary-pbe-elasticity-2025.1.json.gz`, `alexandria-binary-pbe-phonon-2025.1.json.gz`, `wbm-high-energy-states.json.gz` | `get_benchmark_data(...)` — full command in `09_benchmarks/matcalc/submit_matcalc.sh`. matcalc **hardcodes `~/.cache/matcalc`** and ignores `HF_HOME` |
| MLIP-Arena stability | HF `atomind/mlip-arena` → `stability/random-mixture.db` (RM24) | `hf_hub_download(...)` — command in `submit_stability.sh` |
| LiTraj nebDFT2k | 65 MB, 1,656 hops | `python run_litraj_nebdft2k.py --prefetch --cache-dir <DIR>`, then `NEB_CACHE_DIR=<DIR>` |

matcalc's elemental reference tables (`MatPES-PBE-Element-Refs.json.gz`,
`MP-PBE-Element-Refs.json.gz`) ship inside the `matcalc` package — env C covers
them, nothing to fetch.

## 7. MACE-OMAT-0 medium

A public MACE foundation model (`mace-omat-0-medium.model`). Needed for:

- **stage 10.3** (`FEATURE_MODEL`) — the model the latent-space features are
  extracted from, and the model the published analysis used
- **stage 10.1** (`MODEL`) — must be the *same* model as 10.3, or the errors and
  the features describe different things
- the **MatPES aggregate**, as the `OMAT0` entry and as `--ceiling`

You can substitute your own `*_run-1.model` from stage 08 to analyse the subset
model instead — but use one model consistently across 10.1 and 10.3.

> Stage 10.3 needs the **plain** `.model` / `*_run-1.model`, never
> `*_compiled.model`: it walks the module tree to size the descriptor, and a
> TorchScript model has no such attributes. Stage 09's matcalc and matpes are
> the opposite. See `08_compile_model/README.md`.

## 8. The six other MatPES comparison models

`09_benchmarks/matpes/submit_aggregate.sh` aggregates **seven** models: the
Chemenv subset model (v14), MACE-OMAT-0, a DIRECT-selected subset model, and
four random-seed replicas `rdm1`…`rdm4`. The DIRECT and random subsets come
from `04_subset_selection/DIRECT/` and `04_subset_selection/random/`, and their
models from stages 06–08 with the batch sizes of SI Table S2. Neither baseline
selection can be repeated bit for bit from the public data (frame order of the
extxyz conversion for DIRECT, DuckDB reservoir sampling for random; see the
READMEs there), so use the deposited id lists with stage 05 to rebuild the
published training sets.

The four `rdm*` replicas are what define the **noise floor** every significance
statement in the paper tables rests on, and `--aggregate` recognises a replica
purely by its label starting with `rdm` / `seed` / `random`. Mislabel one and
you lose the noise floor with no warning.

If you have fewer models, cut `RUN_DIRS` and `LABELS` down to match — the script
checks that the two arrays are the same length and that every directory exists.

The committed `09_benchmarks/matpes/paper_tables/` are the published output of
the full seven-model run.

## 9. The published subset id list — how to skip 440 node-days

Stages 01–04 cost roughly 440 node-days and need a 400 GB machine. If you want
to *use* the selected subset rather than re-derive it, take the published
`best_subset_*.txt` (795,721 OMat24 material ids), point `SUBSET_IDS` at it and
start at stage 05.

Also published: `env_ids_found_distribution.txt`, the per-token atom counts of
the selected subset (5,463,490 rows summing to 6,566,253 atoms).

> **The archive DOI in `README.md` is still a placeholder.** Until it is filled
> in, this shortcut is not actually available to anyone outside the group, and
> stages 01–04 are the only route to the id list.

---

## Working space

Not input data, but it has to be planned for:

| Stage | Produces | Notes |
|---|---|---|
| 01a | one HDF5 per MPI rank per sub-dataset | written to the **current** directory |
| 01b | 11 parquet files | `parquet_file_collection/` |
| 02 | `freq_chemenv.duckdb` | read by **both** 03 and 04 — keep it |
| 03 | `material_vectors.duckdb` + 1000 batch parquets | `DUCK_TEMP_DIR` can reach several hundred GB |
| 04 | `best_subset_*.txt`, `env_ids_found_distribution.txt`, `restart.npz` | run in an **empty** directory — a stale `restart.npz` silently resumes |
| 06 | 128 + 16 + 16 HDF5 shards + `statistics_fixed.json` | plus `SHARD_DIR`, about one extra copy of each input |
| 10.3 | 8 feature shards | ~12.5 GB, loaded whole by 10.4 |

---

## Known gaps

One thing a reproducer needs that no script here provides:

1. **MatPES → extxyz conversion** with `energy` / `forces` / `stress` keys, and
   rare-earth filtering of the result.

And one behavioural caveat: `.xyz.gz` / `.extxyz.gz` appear in
`input_backend.py`'s suffix list and `check_input.py` accepts them (ASE
decompresses transparently), but the stage-01 tokeniser and stage 05 both read
raw bytes and will fail on a gzipped file. **Decompress before running.**
