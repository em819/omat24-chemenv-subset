# Stage 07 — Train MACE on the selected subset

**Env B** · `gpu` · 2 nodes × 4 A100-40GB = 8 GPUs · 2 days per job (the published run needed two jobs, ~3.8 days)

## Files

`run_mace.py` (shared with stage 08) · `train.yaml` · `submit_train.sh`

## Configure

`train.yaml`, in the `>>> USER CONFIGURATION <<<` block at the top: `name`, the
four data paths, `results_dir` / `log_dir`, and the W&B keys. The data paths are
relative — `../processed_data/...` — so launch the job from a directory that
sits next to your `processed_data/`, or make them absolute. Stage 06 writes
`processed_data/` inside `06_data_preparation/`; see the symlink in its README.

`submit_train.sh` needs `MACE_VENV`; `CONFIG` selects the YAML.

Architecture (matches MACE-OMAT-0):

| Setting | Value |
|---|---|
| `interaction` | `RealAgnosticDensityResidualInteractionBlock` |
| `hidden_irreps` | `128x0e + 128x1o` |
| `num_interactions` / `num_radial_basis` | 2 / 10 |
| `r_max` | 6.0 |
| `distance_transform` / `pair_repulsion` | Agnesi / True |
| `mean` / `std` | 0.0 / 1.0 (overridden by `statistics_fixed.json`, which holds the same values) |

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
sbatch --account=<your allocation> submit_train.sh
```

## Notes

- `wandb` is **`False`** by default, because W&B needs an account and a login
  and a failed login stops the run. The published run used it: set it to `True`
  and fill in your own `wandb_project` / `wandb_name` to match.
- `enable_cueq` requires `cuequivariance` and `cuequivariance-torch` (0.8.1 in
  env B) and the CUDA module. Without them the run still works but is markedly
  slower.

- **The run needs more than one job.** The published run hit the 2-day limit
  after epoch 423 (job 5153255). Resubmitting the same script continued it from
  the last checkpoint (`restart_latest: True`) until early stopping after epoch
  832 (job 5159525). On restart MACE resumes from the saved weights, which are
  the EMA weights, and resets the early-stopping counter.
- **The job ends with an error, and this is expected.** When early stopping
  fires with `distributed: True`, MACE leaves the training loop on rank 0 only,
  and the post-training step fails with `DDP expects same model across all
  ranks`. Training is complete and all checkpoints are written. Stage 08 does
  the export.
- `save_all_checkpoints: True` writes a checkpoint every epoch (~108 MB each,
  ~90 GB for the full run).

## Boundary check

The training log reports `Head 'Default' training dataset size: 795721` and
`avg_num_neighbors 40.5113`. The published model is the **epoch-792**
checkpoint: the epoch with the lowest validation loss (`loss=0.00017116`). With
early stopping this is the last epoch minus `patience` (832 − 40). The last
checkpoint is *not* the best one. To find the best epoch in your own log:

```bash
grep -oE 'Epoch [0-9]+: head: Default, loss=[0-9.e-]+' slurm-*.out | LC_ALL=C sort -t= -k2,2g | head -1
```
