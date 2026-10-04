# Stage 09 — Benchmarks

Four independent benchmarks across three environments. Nothing is shared between
them except the model file.

| Benchmark | Directory | Env | Resources | Model file |
|---|---|---|---|---|
| MatCalc-Bench | `matcalc/` | C | 2 nodes × 4 GPU, 4 h | `_compiled.model` |
| MatPES E/F/S | `matpes/` | B | 1 GPU, 2.5 h (+30 min CPU aggregate) | `_compiled.model` |
| MLIP-Arena stability MD | `stability/` | D | 4 nodes × 4 GPU, 15 h | `_run-1.model` |
| LiTraj nebDFT2k | `nebdft2k/` | D | 4 nodes × 4 GPU, 6 h | `_run-1.model` |

Every path each benchmark needs — the venv, the model, the output directory and
the reference-data cache — is a named variable in the
`>>> USER CONFIGURATION <<<` block at the top of that benchmark's submit script.
The model and the cache have no default. See [`../DATA.md`](../DATA.md) for what
each reference dataset is and where to get it.

Each benchmark fetches its own reference data, and the libraries download on
first use. On a cluster whose compute nodes are offline (MeluXina among them)
that has to happen from a login node first:

| Benchmark | Data | Stage it with |
|---|---|---|
| MatCalc-Bench | HF `materialyze/matcalc-bench` → `~/.cache/matcalc` | `get_benchmark_data(...)` for the four defaults (command in `submit_matcalc.sh`) |
| MatPES | an extxyz you supply | — |
| stability | `atomind/mlip-arena` → `stability/random-mixture.db` | `hf_hub_download(...)` (command in `submit_stability.sh`) |
| nebDFT2k | LiTraj mirror, 65 MB | `python run_litraj_nebdft2k.py --prefetch --cache-dir <DIR>` |

If the cache is missing the scripts warn and continue, so a compute node with
internet just works. `BENCH_OFFLINE=1` makes them stop instead.

After the runs, one summary script per benchmark turns the raw outputs of all
models into the numbers quoted in the paper, and `radar/` combines them:

| Script | Reads | Paper |
|---|---|---|
| `matcalc/summarize_matcalc.py` | `<Bench>_<model>_FULL.csv` of `merge_results.py` | Fig. 2, MatCalc text, SI S4.2 / Fig. S4 |
| `matpes/blowup_attribution.py` | the `*_paper/` directories + the MatPES extxyz | MatPES text on errors > 10 eV/Å and on polymeric nitrogen |
| `stability/summarize_stability.py` | `results/` of `run_stability_bench.py` | Fig. 3, MD-stability text, SI S4.3 |
| `nebdft2k/table_nebdft2k.py` | `results_litraj_nebdft2k/` | Table II, Li-migration text, SI S4.4 |
| `radar/build_radar_metrics.py` | the four summaries + `matpes/paper_tables/` + `*_paper/` | radar figure, Conclusions |

They take one output directory per model with `--runs` and `--labels`; labels
starting with `rdm` / `seed` / `random` are random-seed replicas and are
reported per seed and as the group `random`.

---

## `matcalc/` — MatCalc-Bench

`run_matcalc_bench.py` runs the Equilibrium, Elasticity, Phonon and Softening
tasks; `merge_results.py` collects the per-rank outputs.

```bash
export MATCALC_VENV=/path/to/envC
export MODEL=/path/to/..._compiled.model
sbatch --account=<your allocation> submit_matcalc.sh   # --all-tabulated --dtype float64
```

`ALLOWED_Z = {1..56} ∪ {72..83}` — the model's element coverage; anything else is
dropped and listed.

`EquilibriumBenchmarkHybridRefs` subclasses the stock benchmark and bypasses its
featurisation step (`super(EquilibriumBenchmark, self).run(...)`), which
otherwise crashes on NaN features.

> **Elasticity configuration.** The code comment claims the shear branch was
> reduced from matcalc's defaults. It was not: against matcalc 0.5.1 the strain
> envelopes (±1% normal, ±6% shear) and `fmax` (0.05) are exactly the defaults;
> only the sampling density changed, 4 → 6 points per axis. See
> [`../../PIPELINE.md`](../../PIPELINE.md).
>
> **The fit-quality number is discarded.** `ElasticityCalc` computes
> `r2_score_mean` and warns when it drops below `r2_min=0.95`, but
> `ElasticityBenchmark.process_result` keeps only K and G, so the CSV never sees
> it. Carrying it through would let the negative `G_vrh` outliers be separated
> into model error versus unreliable strain-stress fit — see K2b.

Committed results: `results/`.

`summarize_matcalc.py` (env C) gives, per model and with the random seeds pooled
and per seed, n, excluded % and the median, mean and 5/25/75/95th percentiles of
the formation-energy error, fingerprint distance, softening (1 − scale) and the
signed and absolute percentage errors of K, G and C_V. `--unphysical drop`
(default, as in the paper) removes failed calculations and non-positive moduli
or heat capacities. Run `merge_results.py` for every model first.

---

## `matpes/` — MatPES energy / force / stress

`eval_with_report_efs.py` does both jobs. Run it once per model to produce a
`*_paper/` directory, then once more with `--aggregate` to build the tables.

```bash
export MACE_VENV=/path/to/envB
MODEL=… TESTSET=… OUT=… LABEL=Chemenv sbatch --account=… submit_eval_single.sh
MATPES_RUNS=… sbatch --account=… submit_aggregate.sh
```

`submit_aggregate.sh` lists the paper's seven models in its `RUN_DIRS` /
`LABELS` arrays: the EnvTok model (v14), MACE-OMAT-0 (public), the DIRECT model
and the four random-seed replicas (trained from the subsets of
`04_subset_selection/DIRECT` and `04_subset_selection/random` with stages 06–08).
Cut the two arrays down to the models you actually have; the script checks that
they are the same length and that every directory exists. See
[`../DATA.md`](../DATA.md) §8.

> **Label the random-seed replicas `rdm1`…`rdm4`.** `--aggregate` treats any
> label beginning with `rdm` / `seed` / `random` as a replica defining the noise
> floor, and every significance statement in the paper tables rests on it. A
> mislabelled replica leaves you with no noise floor and no warning.

`--ceiling` points at the same directory as one of the aggregated models. That
is deliberate: a model compared against itself must come out at exactly zero
excess, which is a self-check on the ceiling arithmetic.

Committed results: `paper_tables/`.

`blowup_attribution.py` (env B) counts the atoms with a force error above
10 eV/Å, the share of them in cells that contain a 0D or 1D fragment and in
cells made only of nitrogen, and recomputes the bulk per-atom table (Table I)
without the nitrogen-only cells. It takes the same `*_paper/` directories and
labels as `--aggregate`, `--ceiling` (the reference model's label) and
`--configs` (the MatPES extxyz the evaluations read).

---

## `stability/` — MLIP-Arena stability MD

NVT heating and NPT compression runs. `run_stability_bench.py` vendors
`get_atoms_from_db`, `get_runtime_stats` and `gather_results` from `mlip_arena`
because the released wheel does not export them.

```bash
export ARENA_VENV=/path/to/envD
export MODEL=/path/to/..._run-1.model
sbatch --account=<your allocation> submit_stability.sh   # --dtype float64 --cueq
```

> `build_worklist` puts the same `Atoms` object into the NVT and the NPT list.
> This is harmless: `MD.fn` (mlip-arena 0.2.0) runs on `atoms.copy()` and never
> changes the object it is given, so every run starts from the database structure
> whatever the rank count.

`summarize_stability.py` (env D) classifies the 80 NPT runs of each model as
survived, physically diverged (singular cell matrix or temperature blow-up),
CUDA OOM, other failure or truncated (wall-clock limit), from the
`*-compression.parquet` and the `.failed_` / `.oom_` markers. It reports mean
completion, finished fraction and the time of the first failure with the ramp's
temperature and pressure. Physical divergence is given with and without OOM
runs; the paper's percentages exclude them.

---

## `nebdft2k/` — LiTraj nebDFT2k Li-migration barriers

```bash
export ARENA_VENV=/path/to/envD
export MODEL=/path/to/..._run-1.model
export NEB_CACHE_DIR=<CACHE_DIR>
# login node, once:
python run_litraj_nebdft2k.py --prefetch --cache-dir "$NEB_CACHE_DIR"
sbatch --account=<your allocation> submit_nebdft2k.sh
```

> **This is the protocol of our manuscript (SI Sec. S4), not the original
> LiTraj one.** `--symmetric-endpoints` (both end-points relaxed to 0.05 eV/Å)
> and `--n-steps 10000` deviate from the LiTraj publication (asymmetric
> end-points, `--n-steps 100`), which is why the script's help marks the flag as
> not comparable to LiTraj's Table 4. Deliberate — but any comparison against
> the LiTraj reference table must say so.

> **Only the CONVERGED-only aggregate block is quotable.** The "ALL items" rows
> are numerically meaningless (MAE ~1×10⁸ eV, R² ~ −1.5×10¹⁹) because a few
> non-converged bands return astronomic `em_pred` and nothing is dropped by
> design.

`--tm-policy flag` runs all 1,656 hops and reports the two Hubbard-U splits
separately. The 621-hop U-free split (646 in the full 1,681-hop index; the
25 hops removed by the f-block cut are all U-free) is where the model's training and
nebDFT2k's plain-PBE reference actually agree; quote that as the headline.

`table_nebdft2k.py` (env D) builds Table II (MAE, RMSE, median |error| per
Hubbard-U split on converged hops; random as mean ± SD over the seeds) and the
median signed error of every model and its shift between the two splits.

---

## `radar/` — summary figure

`build_radar_metrics.py` (numpy and pandas only) computes the 12 spokes of the
radar figure from `matcalc_summary.csv`, `stability_npt_summary.csv`,
`nebdft2k_table.csv`, `matpes/paper_tables/` and the per-atom files in the
`*_paper/` directories (non-bulk median force error).
