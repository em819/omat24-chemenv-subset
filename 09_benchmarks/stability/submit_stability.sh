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

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# ARENA_VENV   REQUIRED. Path to the env D virtualenv DIRECTORY -- see
#              envs/envD-mliparena/. e.g. export ARENA_VENV=/opt/envs/envD-mliparena
#
# MODEL        REQUIRED. Path to the MACE model FILE to benchmark. Use the
#              *_run-1.model exported by stage 08 -- NOT *_compiled.model,
#              which MatCalc and MatPES take instead.
#
# MODEL_NAME   Short label; becomes part of the output parquet filenames
#              (<MODEL_NAME>-heating.parquet, <MODEL_NAME>-compression.parquet).
#              Must be the same for the MD step and the aggregate step below.
#
# OUTDIR       Where trajectories, markers and parquet files are written.
#
# HF_CACHE     HuggingFace cache holding the RM24 structure database
#              (atomind/mlip-arena -> stability/random-mixture.db). It must be
#              staged from a machine with internet -- see the comment below.
#              Only the presence check uses this variable; the download location
#              itself follows the standard HF_HOME/HF cache rules.
#
#     sbatch --account=<your allocation> --partition=<gpu partition> submit_stability.sh
# ============================================================================
VENV="${ARENA_VENV:?set ARENA_VENV to the envD-mliparena virtualenv directory -- see envs/envD-mliparena}"
MODEL="${MODEL:?set MODEL to the *_run-1.model exported by stage 08 (NOT _compiled.model)}"
MODEL_NAME="${MODEL_NAME:-mace-omat-medium-subset-v14_T100_a1_suppm_0.01}"
OUTDIR="${OUTDIR:-$SLURM_SUBMIT_DIR/results}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# ============================================================================

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
if [ ! -d "$HF_CACHE" ]; then
    echo "WARNING: $HF_CACHE not found. The RM24 database will be" >&2
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
