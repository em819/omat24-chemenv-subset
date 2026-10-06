# Stage 06 — MACE preprocessing

**Env B** · `cpu` · 128 cores · **no GPU** · 12 h

Converts the extxyz training/validation/test sets into the HDF5 shards MACE
trains from, and computes the dataset statistics.

## Files

| File | Role |
|---|---|
| `preprocess_parallel.py` | fork-safe parallel replacement for `mace.cli.preprocess_data` |
| `preprocess.yaml` | input paths, `r_max`, shard counts, z-table, E0s |
| `convert_stats.py` | `statistics.json` → `statistics_fixed.json` |
| `submit_preprocess.sh` | runs both |

## Configure

In `preprocess.yaml`, in the `>>> USER CONFIGURATION <<<` block at the top:
`train_file`, `valid_file`, `test_file`. All three ship as `CHANGE_ME`
placeholders and `preprocess_parallel.py` refuses to start until you edit them,
naming the key it is still seeing. Below the block: `h5_prefix` (default
`processed_data/`), `r_max: 6.0`, `batch_size: 32`, `num_process: 128`, the
68-entry `atomic_numbers` z-table and the 89-entry `E0s`.

> **You must supply a validation set; the test set is optional.** Neither comes
> out of stages 01–05, and neither is among the 11 OMat24 training sub-datasets.
> The published run used the **official OMat24 validation split** (657,756
> frames) and **MatPES PBE 2025.2** with rare earths removed (323,096 frames) as
> the test set. If you cannot obtain the validation split, hold out part of the
> selected subset for validation — the model will be comparable, the loss curve
> will not be.
>
> The test shards have no effect on the model: stages 07–08 never read them.
> MACE picks up only files ending in `_test.h5` from `test_dir`, and stage 06
> names them `test_<i>.h5`. To skip the test set, delete the `test_file` line.
> The MatPES evaluation is done in stage 09.
>
> Either way, filter rare earths: the z-table is
> {1…56} ∪ {72…83}, and `atomic_numbers` encodes exactly that.
>
> The 89 `E0s` are OMat24's PBE isolated-atom energies. Correct as they stand if
> your data is OMat24; replace them if it is not.

In `submit_preprocess.sh`: `MACE_VENV` (required), `CONFIG`, `SHARD_DIR`,
`READ_PROCS`, `STATS_PROCS`, `TRAIN_SHARDS`.

## Run

```bash
export MACE_VENV=/path/to/envB
sbatch --account=<your allocation> submit_preprocess.sh
```

## Output

`processed_data/{train,val,test}/` (128 + 16 + 16 HDF5 shards) and
`statistics_fixed.json`, written under the directory you submit from (normally
`06_data_preparation/`). Stages 07 and 08 read it as `../processed_data/` from
their own directories, so make it visible there, e.g. from the repository root:
`ln -s 06_data_preparation/processed_data processed_data`.

## Notes

- **No CUDA module, on purpose.** Loading CUDA maps its libraries into all 128
  fork workers; that inflated virtual memory is what exhausts the fork commit
  limit. The Python script also sets `CUDA_VISIBLE_DEVICES=""` before importing
  torch, so it stays CPU-only either way.
- **Sharding is round-robin** (frame *i* → shard *i % N*), which keeps each
  shard compositionally representative.
- `convert_stats.py` does three things: strips `np.float64(...)` wrappers, trims
  `atomic_energies` from the 89-entry superset to the 68-element z-table, and
  pins `mean`/`std` to 0.0/1.0. Without the trim, `mace_run_train` fails at the
  first forward pass with
  `mat1 and mat2 shapes cannot be multiplied (Nx68 and 89x1)`.
- If preprocessing ENOMEMs at fork under strict overcommit, halve
  `READ_PROCS`/`STATS_PROCS` to 64.

## Boundary check

795,721 train + 657,756 val + 323,096 test configurations.
