#!/bin/bash -l
#SBATCH --job-name=mlip-stability
#SBATCH --account=CHANGE_ME            # was p200243                 # <-- your project id
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=0-15:00:00                 # 200 x 10 ps MD is heavier than matcalc; scale as needed
#SBATCH --nodes=4                         # 4 GPUs per node
#SBATCH --ntasks-per-node=4               # one task per A100 on the node
#SBATCH --gres=gpu:4                      # whole node: 4x A100-40GB
#SBATCH --cpus-per-task=32                # 128 cores / 4 tasks
#SBATCH --output=mlip-stability-%j.out
#SBATCH --error=mlip-stability-%j.err
#SBATCH --requeue                         # .done markers let requeued jobs resume

set -euo pipefail

# >>> EDIT <<<
VENV="${ARENA_VENV:?set ARENA_VENV -- see envs/envD-mliparena}"
# <-- your checkpoint (the ONLY thing you normally change between runs)
MODEL="/project/scratch/p200243/evgeny/Projects/OMat24_for_So3lr/active_learning_of_OMat24/4th_generation_tokens_train-mlff/mace-models/v14_T100_alpha1_support_material/be1024_universal_loss_dft_e0s_use_stress_delta_0.01_lr_0.001_num_radial_basis10/compile_model/converged/mace-subset-2nd_gen_chemenv_v14_T100_a1_suppm-omat-medium-be1024_universal_loss_dft_e0s_use_stress_delta_0.01_lr_0.001_rb10_run-1.model"   # <-- your checkpoint
MODEL_NAME="mace-omat-medium-subset-v14_T100_a1_suppm_0.01"
OUTDIR="$SLURM_SUBMIT_DIR/results"

module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source "$VENV/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Reduce CUDA fragmentation OOMs. NPT compression toward 500 GPa shrinks cells and
# grows neighbour lists, so allocations spike and shrink; expandable_segments lets
# the allocator return memory to the pool instead of fragmenting it. The driver also
# tags any run that still OOMs (.oom marker) so aggregation reports it separately
# from a genuine model blow-up.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The RM24 structure database comes from the HF hub. run_stability_bench.py's
# get_atoms_from_db() downloads it on first use, but MeluXina compute nodes have
# no internet, so stage it from a login node:
#
#   python -c "from huggingface_hub import hf_hub_download
#              hf_hub_download(repo_id='atomind/mlip-arena', repo_type='dataset',
#                              subfolder='stability', filename='random-mixture.db')"
#
if [ ! -d "$HOME/.cache/huggingface" ]; then
    echo "WARNING: ~/.cache/huggingface not found. The RM24 database will be" >&2
    echo "         downloaded now; on an offline compute node that will fail." >&2
    echo "         Stage it from a login node first (see the comment above)." >&2
    if [ "${BENCH_OFFLINE:-0}" = "1" ]; then exit 1; fi
else
    export HF_HUB_OFFLINE=1      # cache present -> never reach for the network
fi

# --- Parallel MD step: each task owns a disjoint stride of the NVT+NPT work-list,
# pinned to one GPU. -K0 keeps a hard rank death (OOM/segfault) from tearing down
# the whole step; the driver swallows per-structure failures and exits 0 anyway.
# Call the venv python by ABSOLUTE PATH: `#!/bin/bash -l` sources your profile,
# which activates conda base, which can shadow the venv's `python` on PATH even
# after `source .../activate`. The absolute path is immune to that ordering.
srun --kill-on-bad-exit=0 "$VENV/bin/python" "$SLURM_SUBMIT_DIR/run_stability_bench.py" \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --outdir "$OUTDIR" \
    --dtype float64 \
    --cueq \
    --element-policy supported || true

# --- Aggregation: single process (NOT srun) globs every rank's trajectories into
# the arena parquet files and prints the headline survival numbers.
echo "MD step finished. Aggregating..."
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/run_stability_bench.py" \
    --aggregate \
    --model-name "$MODEL_NAME" \
    --outdir "$OUTDIR"

echo "Done. Parquet files in: $OUTDIR"
echo "  ${MODEL_NAME}-heating.parquet     (NVT 300->3000 K, 10 ps)"
echo "  ${MODEL_NAME}-compression.parquet (NPT + 0->500 GPa, 10 ps)"
echo "  element_flags.csv                 (out-of-support-element tags for slicing)"
