#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --mem=0
#SBATCH --account=CHANGE_ME            # was p200243
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH -J chemenv-latent-eval

# ============================================================================
# Stage 10.1 -- evaluate the model on the reference set and merge the
# predictions with the DFT reference.
# Environment: envB-mace.  Resources: 1 GPU, about 2 h.
#
# Writes all_structures_ref_and_predicted.xyz into --outdir. Stage 10.2 reads it
# as ../all_structures_ref_and_predicted.xyz, so --outdir should be the parent of
# the directory stage 10.2 runs in.
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

time srun python ./evaluate_and_merge.py \
    --configs "${REF_XYZ:?path to the reference extxyz}" \
    --model   "${MODEL:?path to the MACE .model}" \
    --outdir  .. \
    --dtype   float64
