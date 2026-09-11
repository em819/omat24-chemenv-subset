# Stage 09 — Benchmarks

Four independent benchmarks across three environments. Nothing is shared between
them except the model file.

| Benchmark | Directory | Env | Resources | Model file |
|---|---|---|---|---|
| MatCalc-Bench | `matcalc/` | C | 2 nodes × 4 GPU, 4 h | `_compiled.model` |
| MatPES E/F/S | `matpes/` | B | 1 GPU, 2.5 h (+30 min CPU aggregate) | `_compiled.model` |
| MLIP-Arena stability MD | `stability/` | D | 4 nodes × 4 GPU, 15 h | `_run-1.model` |
| LiTraj nebDFT2k | `nebdft2k/` | D | 4 nodes × 4 GPU, 6 h | `_run-1.model` |

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

---

## `matcalc/` — MatCalc-Bench

`run_matcalc_bench.py` runs the Equilibrium, Elasticity, Phonon and Softening
tasks; `merge_results.py` collects the per-rank outputs.

```bash
export MATCALC_VENV=/path/to/envC
sbatch submit_matcalc.sh          # --all-tabulated --dtype float64
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
> `../../CHANGES_FROM_AS_RUN.md` (K2).
>
> **The fit-quality number is discarded.** `ElasticityCalc` computes
> `r2_score_mean` and warns when it drops below `r2_min=0.95`, but
> `ElasticityBenchmark.process_result` keeps only K and G, so the CSV never sees
> it. Carrying it through would let the negative `G_vrh` outliers be separated
> into model error versus unreliable strain-stress fit — see K2b.

Committed results: `results/`.

---

## `matpes/` — MatPES energy / force / stress

`eval_with_report_efs.py` does both jobs. Run it once per model to produce a
`*_paper/` directory, then once more with `--aggregate` to build the tables.

```bash
export MACE_VENV=/path/to/envB
MODEL=… TESTSET=… OUT=… LABEL=Chemenv sbatch submit_eval_single.sh
MATPES_RUNS=… sbatch submit_aggregate.sh
```

> **Label the random-seed replicas `rdm1`…`rdm4`.** `--aggregate` treats any
> label beginning with `rdm` / `seed` / `random` as a replica defining the noise
> floor, and every significance statement in the paper tables rests on it. A
> mislabelled replica leaves you with no noise floor and no warning.

`--ceiling` points at the same directory as one of the aggregated models. That
is deliberate: a model compared against itself must come out at exactly zero
excess, which is a self-check on the ceiling arithmetic.

Committed results: `paper_tables/`.

---

## `stability/` — MLIP-Arena stability MD

NVT heating and NPT compression runs. `run_stability_bench.py` vendors
`get_atoms_from_db`, `get_runtime_stats` and `gather_results` from `mlip_arena`
because the released wheel does not export them.

```bash
export ARENA_VENV=/path/to/envD
sbatch submit_stability.sh        # --dtype float64 --cueq --element-policy supported
```

> **Known issue (K1).** `build_worklist` puts the *same* `Atoms` object into
> both the NVT and NPT lists, and `MD.fn` mutates positions in place. Whether it
> bites depends on the rank count: NVT item *j* runs on rank `j % world` and NPT
> item *j* on rank `(j + n_nvt/2) % world`, so the two ensembles usually land on
> different processes and never notice. Where they land on the same process, the
> NPT run starts from positions the NVT run already moved. **Add `atoms.copy()`
> in `build_worklist` before running at a different node count.**

---

## `nebdft2k/` — LiTraj nebDFT2k Li-migration barriers

```bash
export ARENA_VENV=/path/to/envD
# login node, once:
python run_litraj_nebdft2k.py --prefetch --cache-dir <CACHE_DIR>
sbatch submit_nebdft2k.sh
```

> **This is the sensitivity-check protocol, not the published one.**
> `--symmetric-endpoints` is marked in the script's own help as *"SENSITIVITY
> CHECK ONLY, not paper-comparable"*, and the published LiTraj cap is
> `--n-steps 100` against the 10000 used here. Deliberate — but any comparison
> against the LiTraj reference table must say so.

> **Only the CONVERGED-only aggregate block is quotable.** The "ALL items" rows
> are numerically meaningless (MAE ~1×10⁸ eV, R² ~ −1.5×10¹⁹) because a few
> non-converged bands return astronomic `em_pred` and nothing is dropped by
> design.

`--tm-policy flag` runs all 1,656 hops and reports the two Hubbard-U splits
separately. The 646-hop U-free split is where the model's training and
nebDFT2k's plain-PBE reference actually agree; quote that as the headline.
