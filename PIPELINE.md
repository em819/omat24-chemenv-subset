# Pipeline reference

One section per stage: what it reads, what it writes, how it is launched, and
the record counts observed when the published results were produced. Those
counts are the checkpoints — if your numbers match at each boundary, your run
matches the paper's.

Environment ids (**A**–**D**) are defined in [`envs/README.md`](envs/README.md).
The data each stage reads is catalogued in [`DATA.md`](DATA.md); every path is a
named variable in that script's `>>> USER CONFIGURATION <<<` block.

---

## 00 — Convert OMat24 to extxyz

`00_convert_omat24/convert_omat24.py` (env A, MPI, one array task per
sub-dataset) writes `$OMAT_XYZ_DIR/<subset>/all_structures_<subset>.xyz` from
the unpacked ASE databases and drops structures with rare earths or actinides.
Frames follow the database order. Stage 05 and the baselines in
`04_subset_selection/` read these files; stage 01 reads the databases.

---

## 01 — Chemenv tokenisation of OMat24

**Env A** · `cpu` · 20 nodes × 128 ranks · 2 days per sub-dataset

### 01a `tokenize_omat24.py`

Reads the OMat24 training split as `AseDBDataset` and, for every site in every
structure, builds an interpretable token:

```
<element>_<spin>_<neighbour elements>_<polyhedron>_<f5>_<dmin bin>_<asymmetry bin>
```

Coordination environments come from pymatgen's `LocalGeometryFinder` /
`SimplestChemenvStrategy` with an adaptive two-cutoff neighbour search
(relative cutoffs 1.4 and 1.7, Voronoi solid-angle prominence 0.45). Structure
refinement is disabled (`STRUCTURE_REFINEMENT_NONE`) — this is essential, since
site indices must stay aligned with the input structure.

- **Input:** `$OMAT_DATA_DIR` — the directory holding the 11 OMat24
  sub-datasets, as ASE databases or extxyz. Required; no default.
- **Output:** one HDF5 file per MPI rank, `chemenv_<subset>_rank<N>.hdf5`.
- **Launch:** `sbatch submit_tokenize.sh` (a `0-10` job array; the array index
  becomes `OMAT_DB_INDEX`, which picks the sub-dataset).

> **Token grammar warning.** The field count is *not* fixed: 11 Chemenv symbols
> contain underscores. Always index the trailing fields from the END of the
> token, never from the start. `select_subset.py` and `concat_to_parquet.py`
> both do this; anything you write against these tokens must too.

### 01b `concat_to_parquet.py`

Streams the per-rank HDF5 shards into one parquet file per sub-dataset via
`pq.ParquetWriter`, dropping perturbation-artefact rows matching
`_999_\d+_\d+$`.

- **Input:** the rank HDF5 files; argv = `<subset name> <expected ranks>`.
- **Output:** one parquet per sub-dataset.
- **Launch:** `sbatch submit_concat.sh` (11 concurrent single-rank tasks).

**Boundary check:** 194,928,651 tokens · 1,180,921,883 sites · 64,648,643
materials.

### Reproducibility note: `sym_measure` is not bit-stable

Running the tokeniser twice on identical input gives slightly different
`sym_measure` values for a small number of sites. pymatgen's chemenv uses
random permutation sampling when it tries to match high-coordination sites, so
the continuous symmetry measure it reports for those sites varies between runs.
This is a property of pymatgen, not of this code, and it affects the as-run
scripts in exactly the same way.

It does not appear to reach the tokens. `perturb_scale`, the only token field
derived from `sym_measure`, is a bucketed value with edges at 1, 5 and 10, and
the unstable sites are precisely the ones that fit no geometry well — their CSM
is far above the top edge, so the run-to-run spread never crosses a boundary.

The sites affected are the ones chemenv cannot match to any geometry, so their
symmetry measure is far above the top bucket edge and a small run-to-run shift
cannot move them into a different bucket. If you want to check this on your own
data, compare the `sym_measure` and `env_id` columns between two runs.

If you need `sym_measure` itself to be reproducible — it is stored in the
parquet alongside the bucketed value — treat it as an approximate diagnostic
rather than a stable quantity.

---

## 02 — Token frequency database

**Env A** · `cpu` · 1 core · 1 day

`build_frequency_db.py` loads the parquet files into DuckDB and builds three
tables: `data` (all sites), `chemenv_frequency_table` (token → count), and
`material_elements` (material → element list).

- **Input:** the stage-01 parquet files.
- **Output:** `freq_chemenv.duckdb`.
- **Launch:** `sbatch submit_frequency.sh`.

This database is read by **both** stage 03 and stage 04 — it is a shared input,
not a throwaway intermediate. Keep it.

---

## 03 — Integer token ids and material vectors

**Env A** · `cpu` · 128 cores · 2 days

`build_integer_tokens.py` assigns each distinct token an integer id
(`environment_mapping`), counts occurrences (`env_counts`), writes 1000 batch
parquet files sharded by `hash(mat_id) % 1000`, and assembles `material_vectors`
— for each material, the list of token ids it contains and how often.

Counts are **not** normalised (the normalisation block is deliberately
commented out; selection operates on raw per-material counts).

- **Input:** `freq_chemenv.duckdb` (`$CHEMENV_FREQ_DB`; see the USER
  CONFIGURATION block at the top of the file).
- **Output:** `environment_mapping`, `env_counts`, `material_vectors`, batch parquets.
- **Launch:** `sbatch submit_integer_tokens.sh`.

---

## 04 — Subset selection

**Env A** · `cpu` · 128 cores · **400 GB RAM** · 2 days

`select_subset.py` is the core of the method. It builds *demand classes* from an
iterative coarsening ladder over the token grammar, then runs a lazy-greedy
(CELF) set cover over materials to satisfy every class.

Production settings, in `main()`:

| Knob | Value |
|---|---|
| `coarsening_T` | 100 |
| `coarsening_design` | `'a'` (strict partition) |
| `support_metric` | `'materials'` |
| `n_coverage` | 1 |
| `uniformity_weight` | 0.0 |
| `cost_alpha` | 1.0 |
| `enable_floors` | False |
| `enable_floor_topup` | True |
| `enable_dist_floor` / `enable_asym_floor` | True |
| `enable_full_token_rung` | True |
| `enable_dedup` | False |
| `enable_forced_selection` | True |
| `enable_redundancy_prune` | True |
| `restart_interval` | 50000 |
| `duck_threads` / `duck_memory` | 128 / 400 GB |

The ladder is a lexicographic product of identity levels and a modifier chain,
with the **modifier outer** (this nesting order is what version 14 changed, and
it is what turns wall-atom depletion into enrichment):

```python
_MODIFIER_CHAIN = [("dA", "||'|'||dmin||'|'||a_bin"), ("A", "||'|'||a_bin"), ("", "")]
```

- **Input:** `freq_chemenv.duckdb` and the stage-03 tables
  (`$V10_ENV_FREQ_DB` / `$V10_TOKEN_VEC_DB`; see the USER CONFIGURATION block at
  the top of the file).
- **Output:** `best_subset_*.txt` (one material id per line) and
  `env_ids_found_distribution.txt` (per-token atom counts in the subset).
- **Launch:** `sbatch submit_select_subset.sh`.

> **Run this in its own working directory.** `restart.npz` and
> `env_ids_found_distribution.txt` are fixed filenames; a leftover `restart.npz`
> makes a fresh run silently resume the previous one.

**Boundary checks:** 3,443,621 demand classes at T=100 · 3 forced materials ·
greedy 837,273 → redundancy prune −5.11% → 794,511 → floor top-up +1,210 →
**795,721 materials / 6,566,253 atoms**. All four Approach-F floors 100% present
after top-up; all demand classes satisfied. The per-rung class counts sum
exactly (11 rungs + catch-all = 194,928,651 tokens; 3,443,621 classes).

### 04 baselines — DIRECT and random subsets

- **`04_subset_selection/DIRECT/`** (env E): split the extxyz into 50,000-frame
  chunks → M3GNet `readout` features (512 shards) → stack → PCA (21 PCs) →
  BIRCH (`n` = 116500, threshold 0.1092) → *k* = 4 per cluster → write
  `train.xyz`. **Boundary check:** 64,648,643 rows × 128 · 116,328 clusters ·
  **458,754 structures**. See the README there for the matgl readout weights.
- **`04_subset_selection/random/`** (env A, reads `freq_chemenv.duckdb` from
  stage 02): 35 reserved materials + reservoir sample, one job per seed →
  `random_subset_mat_ids.txt` + `train.xyz`. **Boundary check:** pool
  64,648,643 · 35 reserved · 356,249 selected.

Both write the training extxyz directly, so they skip stage 05.

`04_subset_selection/subset_statistics.py` (env A) compares the training sets:
structure and atom counts and the per-atom DFT force distribution. **Boundary
check:** EnvTok 795,721 structures / 6,566,253 atoms (8.25 atoms per
structure), median |F| 1.144 eV/Å; four random seeds pooled: median 1.135 eV/Å.

---

## 05 — Extract the selected frames

**Env A** · `cpu` · 4 cores · 12 h

`generate_train_set.py` streams the OMat24 extxyz files, matches the `sid=`
token in each comment line against the selected id list, and copies matching
frames through verbatim (no re-parsing, no re-writing of the geometry).

- **Input:** `$SUBSET_IDS` (`best_subset_*.txt`) and `$OMAT_XYZ_DIR` (the
  OMat24 training split as uncompressed extxyz).
- **Output:** `$TRAIN_XYZ`, the subset training extxyz — stage 06's `train_file`.
- **Launch:** `sbatch submit_generate_train_set.sh`.

**Boundary check:** 64,648,643 frames scanned · **795,721 written** ·
795,721/795,721 ids matched (100%). The scanned count equals the material count
from stage 01 exactly, which is what proves nothing was lost or double-counted.

> The extxyz files themselves are produced from the OMat24 ASE databases by
> stage 00. Stage 05 streams raw text and cannot read an ASE database or a
> gzipped file. See [`DATA.md`](DATA.md).

---

## 06 — MACE preprocessing

**Env B** · `cpu` · 128 cores · **no GPU** · 12 h

`preprocess_parallel.py` is a fork-safe parallel replacement for
`mace.cli.preprocess_data`. It sets `CUDA_VISIBLE_DEVICES=""` before importing
torch, splits the input round-robin (frame *i* → shard *i % N*), and reuses
MACE's own `load_from_xyz`, `save_configurations_as_HDF5` and
`pool_compute_stats`.

The SLURM script deliberately loads **no CUDA module**: mapping the CUDA
libraries into all 128 workers is what exhausts the fork commit limit.

`convert_stats.py` then rewrites `statistics.json` → `statistics_fixed.json`:
strips `np.float64(...)` wrappers, trims `atomic_energies` from the 89-entry
superset to the 68-element z-table (otherwise MACE fails at the first forward
pass with `mat1 and mat2 shapes cannot be multiplied (Nx68 and 89x1)`), and pins
`mean`/`std` to 0.0/1.0. `submit_preprocess.sh` chains it automatically.

- **Input:** `preprocess.yaml`. Its three dataset paths ship as `CHANGE_ME`
  placeholders and must be edited (`r_max` 6.0, `batch_size` 32,
  `num_process` 128, 68 atomic numbers, 89 E0s are the published values).
- **Output:** `processed_data/{train,val,test}/` + `statistics_fixed.json`.
- **Launch:** `sbatch submit_preprocess.sh`.

**Boundary check:** 795,721 train + 657,756 val + 323,096 test configurations →
128 + 16 + 16 HDF5 shards.

---

## 07 — Training

**Env B** · `gpu` · 2 nodes × 4 A100 · 2 days

`run_mace.py train.yaml`. Architecture matches MACE-OMAT-0:
`RealAgnosticDensityResidualInteractionBlock`, `hidden_irreps
"128x0e + 128x1o"`, 2 interactions, 10 radial basis, `r_max` 6.0, Agnesi
distance transform, pair repulsion. Universal loss with energy/forces/stress
weights 1/10/10 and `huber_delta` 0.01; AMSGrad at lr 1e-3; EMA 0.999;
batch size 128 per GPU; `enable_cueq`; distributed.

- **Input:** `processed_data/` from stage 06.
- **Output:** checkpoints under `results_dir`/`checkpoints_dir`.
- **Launch:** `sbatch submit_train.sh`.

**Boundary check:** the training log reports
`Head 'Default' training dataset size: 795721` and `avg_num_neighbors 40.5113`.
The published model is the **epoch-792** checkpoint.

---

## 08 — Export the model

**Env B** · `gpu` · 2 nodes × 4 A100 · 1 h

`run_mace.py compile.yaml`. Not a separate program: `compile.yaml` sets
`max_num_epochs` to a value ≤ the checkpoint's epoch, so MACE's training loop is
skipped and `run_train` falls straight through to the cueq→e3nn conversion and
model save.

Copy the checkpoint you want to export into this directory first —
`checkpoints_dir`, `model_dir` and `results_dir` are all `"."` so the live
training run is untouched.

- **Output:** `<name>_compiled.model` and `<name>_run-1.model`.
- **Launch:** `sbatch submit_compile.sh`.

> Stage 09 uses **both**: MatCalc and MatPES take `_compiled.model`, while the
> stability and NEB benchmarks take `_run-1.model`.

---

## 09 — Benchmarks

See [`09_benchmarks/README.md`](09_benchmarks/README.md). Four independent
benchmarks across three environments:

| Benchmark | Env | Resources | Model file |
|---|---|---|---|
| MatCalc-Bench (equilibrium / elasticity / phonon / softening) | C | 8 GPUs, 4 h | `_compiled.model` |
| MatPES E/F/S | B | 1 GPU, 2.5 h (+ 30 min CPU aggregate) | `_compiled.model` |
| MLIP-Arena stability MD | D | 16 GPUs, 15 h | `_run-1.model` |
| LiTraj nebDFT2k barriers | D | 16 GPUs, 6 h | `_run-1.model` |

**Boundary check:** MatPES evaluates 323,096 structures / 3,011,678 atoms —
exactly the stage-06 test count.

### 09 summaries

`matcalc/summarize_matcalc.py`, `matpes/blowup_attribution.py`,
`stability/summarize_stability.py`, `nebdft2k/table_nebdft2k.py` and
`radar/build_radar_metrics.py` turn the raw outputs of all models into the
paper's numbers (see the benchmarks README). **Boundary checks** for
MACE-OMAT-0, the EnvTok model and the four random seeds (the DIRECT values
depend on the DIRECT model used):

- MatCalc: mean softening 2.84 / 4.21 / 3.89 %; formation-energy MAE
  56.0 / 62.7 / 65.5 meV/atom; G median +1.42 / −5.77 / −5.62 %.
- MatPES: atoms with a force error above 10 eV/Å 4150 / 4428 / 2924–6226;
  96.3 % of EnvTok's in cells with a 0D or 1D fragment; 97.2 % of
  MACE-OMAT-0's in nitrogen-only cells.
- Stability: first failure of MACE-OMAT-0 at 3.14 ps; mean completion
  75.4 / 92.1 / 81.2 %; physically diverged (CUDA OOM excluded) 31 / 6 / 86 of
  80 / 80 / 320 runs.
- nebDFT2k: barrier MAE without Hubbard U 0.146 / 0.124 / 0.148 ± 0.037 eV
  (621 hops).

---

## 10 — Chemenv tokens in the MACE latent space

**Execution order is 1 → 2 → 3 → 4, which is why the subdirectories are
numbered.** It is not the order they were created in.

### 10.1 `evaluate_and_merge.py` — Env B · 1 GPU · 2 h

Runs the model over the reference set and merges predictions with the DFT
reference into `all_structures_ref_and_predicted.xyz` (carrying both predicted
and `REF_energy`/`REF_forces`).

### 10.2 `analyze_predictions.py` — Env A · 2560 ranks · 48 h

Re-tokenises those structures and attaches per-site force/energy errors to each
token. Reads `../all_structures_ref_and_predicted.xyz`.

> The tokenisation constants and functions here are **AST-identical** to stage
> 01's (all 10 shared constants, all 7 shared functions: `_covalent_radius`,
> `_distance_ratio_bins`, `_neighbor_keyset`, `select_site_lse`,
> `build_coord_descriptor`, `get_complete_coordination_info`,
> `contains_rare_earths`). That identity is what makes database-side and
> prediction-side tokens the same join key. If you change one, change the other.

### 10.3 `extract_mace_features.py` — Env B · 8 GPUs · 24 h

`get_descriptors(invariants_only=True)` → per-atom invariant descriptors,
D = 256 (128 features × 2 layers), written as shards.

### 10.4 `chemenv_feature_metrics.py` — Env B · 4 GPUs · 2 h

kNN purity, size-matched compactness null, and kNN-graph connected components
per token. Run with `--k 30 --k-graph 15 --var-keep 0.95 --max-dims 50
--min-size 20 --knn-backend faiss-gpu`.

**Boundary check:** 8 shards × ~82,219 = 657,756 structures / 12,019,558 atoms —
the stage-06 validation set.

> **Interpreting the output.** A correlation between token compactness and
> absolute force error is confounded by force magnitude: tokens whose atoms
> carry large forces have large absolute errors regardless of how compact they
> are. The script reports the relative-error correlation alongside it; that is
> the one to read.

### 10.5 `5_descriptor_analysis/` — Env A / B · 1 CPU node

The analyses of SI S5 on the outputs of 10.3, 10.4 and stage 03 (see the
stage-10 README). **Boundary checks:** 18,445 descriptors with at least 30
atoms; 18,153 single-region (98.42 %) and 292 split; 574 diffuse (3.11 %);
neighbours with a different descriptor share the central element in 99.9 %,
the ligand-element set in 69.7 % and the coordination number in 56.9 % of
cases.
