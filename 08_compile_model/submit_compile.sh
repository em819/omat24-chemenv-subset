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
# BEFORE RUNNING: copy exactly ONE checkpoint (.pt) into the directory you
#         submit from -- compile.yaml sets checkpoints_dir to "." and
#         restart_latest picks the newest .pt it finds there.
# OUTPUT: <name>_compiled.model  (used by stage 09 matcalc + matpes)
#         <name>_run-1.model     (used by stage 09 stability + nebdft2k,
#                                 and by stage 10.3 -- which CANNOT take
#                                 the TorchScript _compiled.model)
#
#     sbatch --account=<your allocation> --partition=<gpu partition> submit_compile.sh
# ============================================================================
MACE_VENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
CONFIG="${CONFIG:-./compile.yaml}"
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "$MACE_VENV/bin/activate"

export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1

srun --kill-on-bad-exit=1 --gpu-bind=none python ../07_training/run_mace.py "$CONFIG"
