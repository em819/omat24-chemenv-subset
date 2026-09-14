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

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MACE_VENV  REQUIRED. Path to the env B virtualenv DIRECTORY -- see
#            envs/envB-mace/. e.g. export MACE_VENV=/opt/envs/envB-mace
#
# MODEL      REQUIRED. The MACE model FILE to evaluate: the *_compiled.model
#            exported by stage 08 (MatPES and MatCalc take the compiled export;
#            stability and NEB take *_run-1.model instead).
#
# TESTSET    REQUIRED. The MatPES test set as a single extxyz FILE with DFT
#            energy/forces/stress. The published run used MatPES PBE 2025.2
#            with rare earths removed (323,096 structures) -- the same file
#            stage 06 uses as its test_file. Not produced by this repository;
#            see DATA.md.
#
# OUT        Output extxyz of per-structure predictions. A directory
#            `<OUT without extension>_paper/` is written alongside it, and THAT
#            directory is what submit_aggregate.sh consumes.
#
# LABEL      How this model is named in the aggregate tables.
#            >>> LABELS MATTER <<< --aggregate treats any label starting with
#            rdm / seed / random as a random-seed REPLICA defining the noise
#            floor that every significance statement rests on. Label the
#            replicas rdm1..rdm4; label everything else descriptively
#            (Chemenv, OMAT0, DIRECT, ...). A mislabelled replica leaves you
#            with no noise floor and no warning.
#
#     MODEL=... TESTSET=... OUT=... LABEL=Chemenv \
#         sbatch --account=<your allocation> submit_eval_single.sh
# ============================================================================
MACE_VENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
MODEL="${MODEL:?set MODEL to the *_compiled.model exported by stage 08}"
TESTSET="${TESTSET:?set TESTSET to the MatPES test extxyz, e.g. matpes_pbe_2025.2_no_rare_earth.xyz -- see DATA.md}"
OUT="${OUT:-test_matpes_efs.xyz}"
LABEL="${LABEL:-}"                     # e.g. Chemenv, OMAT0, DIRECT, rdm1..rdm4
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "$MACE_VENV/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

time srun ./eval_with_report_efs.py \
    --configs "$TESTSET" \
    --model   "$MODEL" \
    --output  "$OUT" \
    ${LABEL:+--label "$LABEL"}
