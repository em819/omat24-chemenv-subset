#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=0-01:00:00
#SBATCH --mem=0
#SBATCH --account=CHANGE_ME            # was p200717
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH -J compile-mace-subset

# ============================================================================
# Stage 08 -- export the trained checkpoint to a .model file (no training).
#
# Environment: envB-mace, same as stage 07.
# Resources:   same job shape as stage 07 but ~1 hour (nothing is trained).
#
# Mechanism: compile.yaml sets max_num_epochs <= the checkpoint's epoch, so
# MACE's `while epoch < max_num_epochs` loop is skipped and run_train falls
# straight through to the cueq->e3nn conversion + model save.
#
# Before running: copy the checkpoint you want to export into this directory
# (compile.yaml sets checkpoints_dir/model_dir/results_dir to "."), so the live
# training run of stage 07 is left untouched.
#
# Uses the shared launcher from stage 07 -- see 07_training/run_mace.py.
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv}/bin/activate"

export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1

srun --kill-on-bad-exit=1 --gpu-bind=none python ../07_training/run_mace.py ./compile.yaml
