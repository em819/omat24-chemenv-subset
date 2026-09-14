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

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MACE_VENV  REQUIRED. Path to the env B virtualenv DIRECTORY -- see
#            envs/envB-mace/.
#
# REF_XYZ    REQUIRED. The reference extxyz FILE with DFT energies and forces
#            that the model is evaluated on. The published run used the same
#            file stage 06 uses as valid_file: the official OMat24 validation
#            split, rare earths removed (657,756 structures). Everything
#            downstream in stage 10 is sized by it. See DATA.md.
#
# MODEL      REQUIRED. The MACE .model FILE to evaluate. The published analysis
#            used MACE-OMAT-0 medium (mace-omat-0-medium.model). USE THE SAME
#            MODEL HERE AND IN STAGE 10.3 -- the features and the errors must
#            come from one model to be comparable.
#
# OUTDIR     Where predictions.xyz and all_structures_ref_and_predicted.xyz are
#            written. Stage 10.2 reads the merged file as
#            ../all_structures_ref_and_predicted.xyz, so this must be the PARENT
#            of the directory stage 10.2 runs in. Default "..", i.e. the
#            10_latent_space_analysis/ directory.
#
#     REF_XYZ=... MODEL=... sbatch --account=<your allocation> submit_evaluate.sh
# ============================================================================
MACE_VENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
REF_XYZ="${REF_XYZ:?set REF_XYZ to the reference extxyz with DFT energies/forces -- see DATA.md}"
MODEL="${MODEL:?set MODEL to the MACE .model file to evaluate (same model as stage 10.3)}"
OUTDIR="${OUTDIR:-..}"
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "$MACE_VENV/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

time srun python ./evaluate_and_merge.py \
    --configs "$REF_XYZ" \
    --model   "$MODEL" \
    --outdir  "$OUTDIR" \
    --dtype   float64
