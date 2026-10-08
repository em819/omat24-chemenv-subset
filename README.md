# Subset selection from OMat24 with interpretable local chemical-environment descriptors

Code accompanying the manuscript *Atoms that Matter: Largest Materials Datasets
Are 99% Redundant, and Local Chemistry Can Guide the Required 1%* by
E. Moerman, I. Poltavsky, A. Charkin-Gorbulin and A. Tkatchenko.

This repository contains the complete workflow used to (1) tokenise every local
chemical environment in the OMat24 training split into interpretable EnvTok
descriptors (called Chemenv tokens in the code), (2) select a minimal training
subset that covers the resulting descriptor space, (3) train a MACE machine-learned force field on that subset,
(4) benchmark it, and (5) analyse how the descriptor tokens map onto the
model's latent space.

**Scripts only.** Datasets, model checkpoints and benchmark trajectories are not
in git — see [Data availability](#data-availability).

---

## The pipeline at a glance

```
OMat24 ASE DBs ──── 00  convert to extxyz, drop rare earths (read by 05 and the 04 baselines)
   │
   │  01  tokenise every site               2560 MPI ranks × 2 d × 11 sub-datasets
   ▼
per-site Chemenv tokens (parquet)
   │  02  frequency database                128 cores × 1 d
   │  03  integer token ids + material vectors   128 cores × 2 d
   ▼
   │  04  coverage-driven subset selection   128 cores × 2 d, 400 GB
   │      (baselines: 04/DIRECT, 04/random → their own train.xyz)
   ▼
795,721 material ids
   │  05  extract those frames to extxyz     4 cores × 12 h
   │  06  MACE preprocessing → HDF5          128 cores × 12 h
   │  07  train MACE                         8 × A100 × 2 d
   │  08  export checkpoint → .model         8 × A100 × 1 h
   ▼
trained MLFF ──┬── 09  benchmarks: MatCalc-Bench, MatPES, MLIP-Arena stability, LiTraj nebDFT2k
               └── 10  Chemenv tokens in MACE latent space
```

Each numbered directory is one stage, contains the script(s) for that stage, the
SLURM script that ran them, and a `README.md` with inputs, outputs and the
record counts observed at that stage boundary. Start with
[`PIPELINE.md`](PIPELINE.md) for the full stage-by-stage I/O contract, and
[`DATA.md`](DATA.md) for what you have to download before any of it runs.

## Before you start: this is expensive

Stage 01 alone is **20 CPU nodes × 2 days × 11 sub-datasets ≈ 440 node-days**.
Stage 04 needs a single node with **400 GB of RAM**. Training is 8 A100s for two
days. Nothing here runs on a laptop.

If your goal is to *use* the selected subset rather than re-derive it, skip
stages 01–04 entirely and take the published list of 795,721 OMat24 material ids
from the data archive (see below); stage 05 turns that list into a training set.

## Compute requirements per stage

| Stage | Script | Partition | Nodes × tasks | GPUs | Walltime | Env |
|---|---|---|---|---|---|---|
| 00 | `convert_omat24.py` ×11 | cpu | 5 × 128 = 640 ranks | — | 2 d each | A |
| 01a | `tokenize_omat24.py` ×11 | cpu | 20 × 128 = 2560 ranks | — | 2 d each | A |
| 01b | `concat_to_parquet.py` | cpu | 1 × 11 | — | 1 d | A |
| 02 | `build_frequency_db.py` | cpu | 1 × 1 (128 c) | — | 1 d | A |
| 03 | `build_integer_tokens.py` | cpu | 1 × 1 (128 c) | — | 2 d | A |
| 04 | `select_subset.py` | cpu | 1 × 1 (128 c, 400 GB) | — | 2 d | A |
| 04/DIRECT | `featurize_m3gnet.py` | cpu | 4 × 128 | — | 3 h | E |
| 04/DIRECT | `direct_sampling.py` | cpu | 1 × 1 (128 c, ~500 GB) | — | 6 h | E |
| 04/random | `select_random_subset.py` (per seed) | cpu | 1 × 1 (128 c) | — | < 1 d | A |
| 05 | `generate_train_set.py` | cpu | 1 × 1 (4 c) | — | 12 h | A |
| 06 | `preprocess_parallel.py` | cpu | 1 × 1 (128 c) | 0 | 12 h | B |
| 07 | `run_mace.py train.yaml` | gpu | 2 × 4 | 8 | 2 d | B |
| 08 | `run_mace.py compile.yaml` | gpu | 2 × 4 | 8 | 1 h | B |
| 09 | `matcalc/run_matcalc_bench.py` | gpu | 2 × 4 | 8 | 4 h | C |
| 09 | `matpes/…` (single model) | gpu | 1 × 1 | 1 | 2.5 h | B |
| 09 | `matpes/…` (aggregate) | cpu | 1 × 1 (16 c) | — | 30 min | B |
| 09 | `stability/run_stability_bench.py` | gpu | 4 × 4 | 16 | 15 h | D |
| 09 | `nebdft2k/run_litraj_nebdft2k.py` | gpu | 4 × 4 | 16 | 6 h | D |
| 10.1 | `evaluate_and_merge.py` | gpu | 1 × 1 | 1 | 2 h | B |
| 10.2 | `analyze_predictions.py` | cpu | 20 × 128 = 2560 ranks | — | 48 h | A |
| 10.3 | `extract_mace_features.py` | gpu | 2 × 4 | 8 | 24 h | B |
| 10.4 | `chemenv_feature_metrics.py` | gpu | 1 × 4 | 4 | 2 h | B |
| 10.5 | `extract_train_coverage.py` | cpu | 1 × 1 (128 c) | — | 2 h | A |
| 10.5 | `neighbour_composition.py` | gpu | 1 × 1 (32 c) | 1 | 1 h | B |

All SLURM scripts were written for [MeluXina](https://docs.lxp.lu/) (EasyBuild
modules, `cpu` / `gpu` partitions, 128-core CPU nodes, 4× A100-40GB GPU nodes).
On another cluster you will need to adjust partitions, accounts and module names.

## Environments

Four environments are needed, plus env E for the DIRECT baseline; `envs/` records each one. See
[`envs/README.md`](envs/README.md) for which stage uses which, how to rebuild
them, and the EasyBuild modules that must be loaded alongside.

| Id | Used by | Python | Key packages |
|---|---|---|---|
| **A** `envA-chemenv` | 01–05, 10.2, 10.5 (part) | 3.10.13 | pymatgen 2025.6.14, mpi4py 4.1.0, h5py 3.14.0, duckdb 1.4.2, numba 0.61.2, fairchem-core 2.3.0 |
| **B** `envB-mace` | 06–08, 09/matpes, 10.1/10.3/10.4, 10.5 (part) | 3.12.12 | mace-torch 0.3.14, torch 2.5.0, cuequivariance 0.8.1 |
| **C** `envC-matcalc` | 09/matcalc | 3.12.3 | matcalc 0.5.1, mace-torch 0.3.16, pymatgen 2026.5.4, phonopy 4.1.0 |
| **D** `envD-mliparena` | 09/stability, 09/nebdft2k | 3.12.3 | mlip-arena 0.2.0, litraj 0.1, mace-torch 0.3.16 |
| **E** `envE-direct` | 04/DIRECT | 3.11.13 | matgl 4.0.3, torch 2.4.1 (CPU), dgl 2.5.0, scikit-learn 1.9.0 |

Every dependency resolves to a released version — on PyPI, or for torch (envs C,
D, E) and dgl (env E) on the wheel index named at the top of that env's
`requirements.txt`. There are no git or local installs to chase.

## Running it somewhere other than MeluXina

**Read [`DATA.md`](DATA.md) first** — what data you need, where each item comes
from, and which variable points at it. Then
[`RUNNING_ELSEWHERE.md`](RUNNING_ELSEWHERE.md) for the directory layout stages
01–04 assume and the full variable reference.

Every script carries a `>>> USER CONFIGURATION <<<` block at the top listing
every path and setting you have to supply, what each one should point at, and
what the stage writes. **No absolute path from the original cluster remains
anywhere**, and an input that cannot be guessed has no default: the job stops
immediately, naming the variable, instead of silently reading the wrong thing.

The short version:

1. `#SBATCH --account=CHANGE_ME` → your allocation, in every submit script
   (a SLURM directive cannot read a shell variable). Or
   `sbatch --account=… --partition=…` at submit time.
2. Environments: conda stages take a name, venv stages a path:
   ```bash
   export CHEMENV_CONDA_ENV=chemenv        # env A; default: active_learning_omat24_python310
   export MACE_VENV=/path/to/envB
   export MATCALC_VENV=/path/to/envC
   export ARENA_VENV=/path/to/envD
   ```
3. Input data — stage 01 reads OMat24 ASE databases or extxyz, auto-detected
   per sub-dataset:
   ```bash
   export OMAT_DATA_DIR=/path/to/omat24
   ```
4. Before committing any compute, verify your files carry what the tokeniser
   needs — `sid` and a periodic cell:
   ```bash
   python 01_token_construction/check_input.py --data-dir /path/to/omat24
   ```

SLURM is not required — the submit scripts are wrappers around plain
`mpirun` / `python`.

Three things cannot be reached from the 11 training sub-datasets alone: stage
06's validation and test sets, stage 09's four benchmark reference datasets, and
the six comparison models the full MatPES table aggregates. See
[`DATA.md`](DATA.md).

## Data availability

Not in git. Deposited at *(Zenodo DOI — fill in)*:

- **`best_subset_*.txt`** — the 795,721 selected OMat24 material ids. The single
  most reusable artefact here: it reproduces the training set without repeating
  the 440 node-days of stages 01–04.
- **`env_ids_found_distribution.txt`** — the per-token atom counts of the
  selected subset (5,463,490 rows, summing to 6,566,253 atoms).
- **The DIRECT and the four random id lists** (458,754 and 356,247–356,249
  OMat24 ids). Neither selection can be repeated bit for bit from the public
  data (frame order of the extxyz conversion for DIRECT, DuckDB reservoir
  sampling for random; see the READMEs in `04_subset_selection/`), so the id
  lists are what reproduces the baseline training sets.
- Benchmark trajectories and per-structure outputs (~5 GB).

Committed, because they are small and are the numbers quoted in the paper:

- `09_benchmarks/matpes/paper_tables/` — the MatPES aggregate tables.
- `09_benchmarks/matcalc/results/` — the MatCalc-Bench per-task results.

The OMat24 dataset itself is from Meta FAIR
([fair-chem/OMat24](https://huggingface.co/datasets/fairchem/OMAT24)); MatPES,
MLIP-Arena and LiTraj nebDFT2k are obtained by their own tooling — see the
stage-09 READMEs.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) for the code in this repository. Third-party datasets and
packages keep their own licenses.
