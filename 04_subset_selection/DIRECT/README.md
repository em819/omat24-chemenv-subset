# Stage 04 (baseline) — DIRECT subset

**Env E** (steps 2–6) · `cpu` · featurisation 4 nodes × 128 cores ·
selection 1 node, ~500 GB

DIRECT (Qi *et al.*, npj Comput. Mater. **10**, 43 (2024)) applied to the OMat24
training split: M3GNet features → PCA (Kaiser's rule) → BIRCH → *k* structures
per cluster. Independent of stages 01–04 of the EnvTok pipeline; it reads the
same OMat24 extxyz files as stage 05. The split step drops structures with rare
earths or actinides, as stage 01 does.

## Files

| Step | Script | Submit | Output |
|---|---|---|---|
| 1 split | `split_xyz_shards.py` | `submit_split.sh` (array 0–10) | `chunks/all_structures_<subset>_NNNNN.xyz` |
| 2 featurise | `featurize_m3gnet.py` | `submit_featurize.sh` (array 0–3) | `feats/shard_<G>.npy`, `.ids.json` |
| 3 stack | `stack_shards.py` | `submit_select.sh` | `feats.npy`, `feats.ids.json` |
| 4–5 select | `direct_sampling.py` | `submit_select.sh` | `selected_460k.{indices.npy,indices.json,ids.json}` |
| 6 write xyz | `generate_train_set_direct.py` | `submit_generate_train_set_direct.sh` | `subset_DIRECT_train.xyz` → stage 06 `train_file` |

Every submit script has a `>>> USER CONFIGURATION <<<` block. Run all steps
from this directory.

```bash
export OMAT_XYZ_DIR=/path/to/omat24_extxyz     # as for stage 05
export DIRECT_VENV=/path/to/envE               # see envs/envE-direct
sbatch --account=<acct> submit_split.sh
sbatch --account=<acct> submit_featurize.sh    # after the split
sbatch --account=<acct> submit_select.sh       # after the featurisation
sbatch --account=<acct> submit_generate_train_set_direct.sh
```

Fetch the M3GNet model once on a machine with internet access (featurisation
runs with `HF_HUB_OFFLINE=1`):

```bash
python -c "import matgl; matgl.load_model('M3GNet-Eform-MP-2018.6.1')"
```

## Settings of the paper

| | |
|---|---|
| encoder | `M3GNet-Eform-MP-2018.6.1`, `readout` feature, 128 dims, CPU, all checkpoint weights loaded (see below) |
| chunks | 50,000 frames → 1,298 chunks; 512 file-mode shards |
| PCA | 21 components (eigenvalue > 1), 81.96 % of the variance, weighted by explained variance |
| BIRCH | `n = 116500`; radius threshold 0.1092 found by `_auto_threshold` (maml starts from 0.5) |
| selection | `--target-size 460000` → *k* = 4 → **458,754 structures / 6,532,238 atoms** |

## Boundary checks (from the paper's run)

- featurisation: every `logs/feat_shard_*.out` contains
  `checkpoint verified: 111 tensors; 12 restored` and a `[done]` line
- stack: `512 shards, 64648643 rows x 128 dims`
- threshold search: `0.08624 → 464668`, `0.1092 → 119183` subclusters
  (target band [116500, 174750])
- clustering: 116,328 non-empty clusters, sizes min 1 / median 184 / mean 555.7 / max 37,234
- `chosen k=4: 458754 structures (-0.27% vs target 460000)`

## Order matters

BIRCH depends on the order of its input rows. The row order of `feats.npy` is
fixed by the sorted chunk names and by `NSHARDS` (shard *G* reads chunks
`[G::NSHARDS]`, and shards are stacked in order). Keep the chunk prefix
`all_structures_<subset>` (`submit_split.sh` sets it regardless of your file
names), 50,000 frames per chunk and `NSHARDS = 512`.

The frame order inside each source extxyz enters the same way. The paper's
extxyz files were written from the OMat24 ASE databases by a parallel MPI
conversion, whose frame order depends on the task scheduling of that run. A new
conversion gives a different order, so re-running these scripts gives a DIRECT
subset made the same way but not the identical one. To rebuild the published
DIRECT training set, use its deposited id list with stage 05.

## Checks

```bash
python direct_sampling.py --self-test        # step 5 against a brute-force reference
python direct_sampling.py --verify-vs-maml   # identical selection to maml (pip install maml)
```

## matgl readout weights

The `M3GNet-Eform-MP-2018.6.1` checkpoint stores its Set2Set readout as
`model.readout.set2set.lstm.*`. matgl 3.0.5 and 4.x name these parameters
`readout.lstm.*`, skip them when loading and leave the 12 LSTM tensors randomly
initialised, without a warning. The `readout` feature then depends on the
random draw of each process. `featurize_m3gnet.py` loads these tensors under the
new names and stops with an error unless every model tensor equals the
checkpoint. The features are then the same as those of
`maml.describers.MatGLStructure` with matgl 3.0.3, which loads the checkpoint
completely.
