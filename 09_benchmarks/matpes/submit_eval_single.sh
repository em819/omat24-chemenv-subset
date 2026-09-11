#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH --time=0-02:30:00
#SBATCH --mem=0
#SBATCH --account=CHANGE_ME            # was p200243
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH -J matpes-eval

# ============================================================================
# Stage 09 / MatPES -- per-structure E/F/S evaluation of ONE model.
#
# Environment: envB-mace.  Resources: 1 GPU, ~2.5 h for 323,096 structures.
#
# Run this once per model you want in the comparison. Each run writes a
# `<output without extension>_paper/` directory; submit_aggregate.sh then takes
# those directories by name, so there is nothing to collect centrally.
#
# LABELS MATTER. --aggregate treats a label starting with rdm / seed / random
# as one of the random-seed replicas that define the noise floor, and every
# significance statement in the paper tables rests on that floor. Label the
# replicas rdm1..rdm4 and everything else descriptively.
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:?path to the compiled .model}"
TESTSET="${TESTSET:?path to matpes_pbe_2025.2_no_rare_earth.xyz}"
OUT="${OUT:-test_matpes_efs.xyz}"
LABEL="${LABEL:-}"                     # e.g. Chemenv, OMAT0, DIRECT, rdm1..rdm4

time srun ./eval_with_report_efs.py \
    --configs "$TESTSET" \
    --model   "$MODEL" \
    --output  "$OUT" \
    ${LABEL:+--label "$LABEL"}
