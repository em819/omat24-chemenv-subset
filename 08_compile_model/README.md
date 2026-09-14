# Stage 08 — Export the checkpoint to a `.model` file

**Env B** · `gpu` · 2 nodes × 4 A100 · 1 h

## Files

`compile.yaml` · `submit_compile.sh` (uses `../07_training/run_mace.py`)

## How it works

This is not a separate program. `compile.yaml` sets `max_num_epochs` to a value
≤ the checkpoint's epoch, so MACE's `while epoch < max_num_epochs` loop is
skipped entirely and `run_train` falls straight through to the cueq→e3nn
conversion, model save and final evaluation.

## Before running

Copy the checkpoint you want to export into this directory. `checkpoints_dir`,
`model_dir` and `results_dir` are all `"."`, which keeps the live stage-07
training run untouched, and `restart_latest: True` picks up the newest `.pt`
it finds there — so put exactly one in.

## Run

```bash
export MACE_VENV=/path/to/envB
sbatch --account=<your allocation> submit_compile.sh
```

`name` and `max_num_epochs` are the two keys you must check — see the
`>>> USER CONFIGURATION <<<` block at the top of `compile.yaml`. `name` must
match `train.yaml`'s, and `max_num_epochs` must be ≤ the epoch of the checkpoint
you are exporting.

## Output

`<name>_compiled.model` and `<name>_run-1.model`.

> **Both exports are used downstream.** MatCalc-Bench and MatPES take
> `_compiled.model`; the MLIP-Arena stability and LiTraj NEB benchmarks, and
> stage 10.3's feature extraction, take `_run-1.model`. Stage 10.3 in particular
> inspects the module tree and cannot accept a TorchScript-compiled model.

## Notes

- `enable_cueq` **must stay True**: the checkpoint's weights are in cueq layout,
  and MACE converts them to e3nn on save.
- `default_dtype` must match training (float32).
- `max_num_epochs: 792` is tied to the published epoch-792 checkpoint. If you
  export a different checkpoint, set this to that epoch or lower.
