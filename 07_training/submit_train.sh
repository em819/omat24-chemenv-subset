#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --mem=0
#SBATCH --account=CHANGE_ME            # was p200717
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH -J train-mace-subset

# ============================================================================
# Stage 07 -- train MACE on the Chemenv-selected subset.
#
# Environment: envB-mace (uv venv, Python 3.12, mace-torch 0.3.14, torch 2.5.0,
#              cuequivariance 0.8.1)
# Resources:   2 GPU nodes x 4 A100 = 8 GPUs, 2 days.
# Data:        the HDF5 shards + statistics_fixed.json from stage 06.
#              train.yaml refers to them as ../processed_data/ -- run this job
#              from a directory that sits next to your processed_data/.
# ============================================================================

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MACE_VENV   REQUIRED. Path to the env B virtualenv DIRECTORY (the one whose
#             bin/activate you source) -- see envs/envB-mace/.
#             e.g. export MACE_VENV=/opt/envs/envB-mace
#
# CONFIG      The MACE config for this stage. Its data paths point at stage 06's
#             processed_data/ RELATIVELY, so launch this job from a directory
#             sitting next to processed_data/ -- or edit the paths in the config.
#
# OUTPUT: checkpoints and logs under train.yaml's results_dir / log_dir.
#         The published model is the epoch-792 checkpoint.
#
#     sbatch --account=<your allocation> --partition=<gpu partition> submit_train.sh
# ============================================================================
MACE_VENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
CONFIG="${CONFIG:-./train.yaml}"
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "$MACE_VENV/bin/activate"

export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1

srun --kill-on-bad-exit=1 --gpu-bind=none python ./run_mace.py "$CONFIG"
