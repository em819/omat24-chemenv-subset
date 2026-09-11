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

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv}/bin/activate"

export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1

srun --kill-on-bad-exit=1 --gpu-bind=none python ./run_mace.py ./train.yaml
