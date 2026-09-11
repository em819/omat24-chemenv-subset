# Stage 07 — Train MACE on the selected subset

**Env B** · `gpu` · 2 nodes × 4 A100-40GB = 8 GPUs · 2 days

## Files

`run_mace.py` (shared with stage 08) · `train.yaml` · `submit_train.sh`

## Configure

`train.yaml`. The data paths are relative — `../processed_data/...` — so launch
the job from a directory that sits next to your `processed_data/`.

Architecture (matches MACE-OMAT-0):

| Setting | Value |
|---|---|
| `interaction` | `RealAgnosticDensityResidualInteractionBlock` |
| `hidden_irreps` | `128x0e + 128x1o` |
| `num_interactions` / `num_radial_basis` | 2 / 10 |
| `r_max` | 6.0 |
| `distance_transform` / `pair_repulsion` | Agnesi / True |
| `mean` / `std` | 0.0 / 1.0 |

Training:

| Setting | Value |
|---|---|
| `loss` | `universal` |
| energy / forces / stress weights | 1 / 10 / 10 |
| `huber_delta` | 0.01 |
| `lr`, `amsgrad`, `weight_decay`, `clip_grad` | 1e-3, True, 1e-6, 100 |
| `ema_decay` | 0.999 |
| `batch_size` (per GPU) | 128 |
| `max_num_epochs` / `patience` | 1000 / 40 |
| `default_dtype` | float32 |
| `enable_cueq` / `distributed` | True / True |

## Run

```bash
export MACE_VENV=/path/to/envB
sbatch submit_train.sh
```

## Notes

- `wandb: True` in `train.yaml`. Set it to `False` or configure your own project
  before running.
- `enable_cueq` requires `cuequivariance` and `cuequivariance-torch` (0.8.1 in
  env B) and the CUDA module. Without them the run still works but is markedly
  slower.

## Boundary check

The training log reports `Head 'Default' training dataset size: 795721` and
`avg_num_neighbors 40.5113`. The published model is the **epoch-792**
checkpoint.
