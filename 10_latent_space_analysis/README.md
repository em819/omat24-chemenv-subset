# Stage 10 — Chemenv tokens in the MACE latent space

Asks whether the interpretable tokens correspond to structure in the model's
learned representation: do atoms sharing a token occupy a compact, pure region
of MACE's invariant descriptor space?

> **Execution order is 1 → 2 → 3 → 4.** That is why the subdirectories are
> numbered; it is not the order they were written in, and alphabetical order
> gets it wrong.

| Step | Directory | Env | Resources |
|---|---|---|---|
| 10.1 | `1_evaluate_mlff/` | B | 1 GPU, 2 h |
| 10.2 | `2_analyze_chemenvs/` | **A** | 20 CPU nodes × 128 ranks, 48 h |
| 10.3 | `3_extract_features/` | B | 2 nodes × 4 GPU, 24 h |
| 10.4 | `4_cluster_metrics/` | B | 1 node × 4 GPU, 2 h |

Note the environment switch at step 10.2 — it is pure pymatgen/MPI, not MACE.

---

## 10.1 `evaluate_and_merge.py`

Runs the model over the reference set and merges predictions with the DFT
reference into one file carrying both.

```bash
export MACE_VENV=/path/to/envB
REF_XYZ=… MODEL=… sbatch --account=<your allocation> submit_evaluate.sh
```

`REF_XYZ` is the reference extxyz with DFT energies and forces — the published
run used the same file stage 06 uses as `valid_file`. `MODEL` is the model whose
predictions are analysed; the published run used MACE-OMAT-0 medium. **Use the
same model here and in 10.3**, or the errors and the features describe different
things. See [`../DATA.md`](../DATA.md).

**Output:** `all_structures_ref_and_predicted.xyz` (predicted energy/forces plus
`REF_energy` / `REF_forces`), written to `--outdir`. Step 10.2 reads it as
`../all_structures_ref_and_predicted.xyz`, so point `--outdir` at the parent of
the directory you will launch 10.2 from.

This was two scripts as run (`eval.py` + `paste_prediction_with_ref_xyz.py`);
see [`../PIPELINE.md`](../PIPELINE.md).

## 10.2 `analyze_predictions.py`

Re-tokenises those structures and attaches per-site force and energy errors to
each Chemenv token. `CHUNK_SIZE = 100` structures per MPI task.

```bash
sbatch --account=<your allocation> submit_analyze.sh   # env A, 2560 ranks
```

**Output:** `all_structures_with_errors_and_v4chemenvs_*.xyz`.

> The tokenisation constants and functions here are **AST-identical** to stage
> 01's — all 10 shared constants and all 7 shared functions (`_covalent_radius`,
> `_distance_ratio_bins`, `_neighbor_keyset`, `select_site_lse`,
> `build_coord_descriptor`, `get_complete_coordination_info`,
> `contains_rare_earths`). That identity is exactly what makes the
> database-side and prediction-side tokens the same join key. **If you edit one,
> edit the other**, and re-check the identity.

## 10.3 `extract_mace_features.py`

Per-atom invariant descriptors via `get_descriptors(invariants_only=True)` /
`extract_invariant`. D = 256 (128 features × 2 layers).

```bash
XYZ=… FEATURE_MODEL=… sbatch --account=<your allocation> submit_extract_features.sh
```

> **Use the plain `.model`, not `_compiled.model`.** This step reaches into the
> module tree (`model.products[0].linear.irreps_out`) to size the invariant
> descriptor; a TorchScript-compiled model has no such attributes and fails with
> a bare `NotImplementedError`. Stage 09 is the opposite for matcalc and MatPES —
> see `08_compile_model/README.md`.

**Output:** 8 shards of per-atom descriptors.

> **Without cueq, use `DTYPE=float32`.** `--dtype float64 --enable-cueq` is the
> published configuration; without cuequivariance installed (on CPU, or a GPU
> box lacking it) float64 can fail inside e3nn's compiled tensor products with
> `RuntimeError: both inputs should have same dtype`. Set `DTYPE=float32
> CUEQ_FLAG=""` — that is the script's own default and it runs cleanly.

## 10.4 `chemenv_feature_metrics.py`

Per-token kNN purity, compactness against a size-matched null, and connected
components of the kNN graph.

```bash
FEAT_DIR=… sbatch --account=<your allocation> submit_metrics.sh
# --k 30 --k-graph 15 --var-keep 0.95 --max-dims 50 --min-size 20
# --knn-backend faiss-gpu
```

Without a GPU set `KNN_BACKEND=ckdtree` (choices: `auto`, `faiss`, `faiss-gpu`,
`ckdtree`). The results are identical; only the speed differs.

---

## Boundary check

8 shards × ~82,219 = **657,756 structures / 12,019,558 atoms** — the stage-06
validation set.

## Interpreting the result

A correlation between a token's compactness in latent space and its **absolute**
force error is confounded by force magnitude: atoms that carry large forces have
large absolute errors whatever their environment looks like. The script reports
the correlation against **relative** force error as well, and that is the one to
read.
