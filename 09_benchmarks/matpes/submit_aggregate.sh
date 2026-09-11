#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:30:00
#SBATCH --mem=0
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J matpes-aggregate

# ============================================================================
# Stage 09 / MatPES -- build the paper tables from several single-model runs.
#
# Environment: envB-mace.  Resources: CPU only, 16 cores, ~30 min.
# No GPU: this step only reads the per-structure NPZ/CSV files that
# submit_eval_single.sh already wrote.
#
# THE PAPER RUN used seven models: the Chemenv subset model (v14), MACE-OMAT-0,
# a DIRECT-selected subset, and four random-seed replicas (rdm1..rdm4) whose
# spread defines the noise floor. Only the v14 directory is reproducible from
# this repository -- the other six models are outside its scope -- so the list
# below is a template. Every entry is a `*_paper` directory produced by
# submit_eval_single.sh.
#
# --ceiling points at the MACE-OMAT-0 directory, i.e. the same path as the
# OMAT0 entry. That is deliberate: it makes OMAT0's own excess_median come out
# as exactly 0.000 in every domain, which is a self-check on the ceiling
# arithmetic. Verified 0.000 in the shipped paper_tables/.
# ============================================================================

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv}/bin/activate"

# Directory holding the *_paper output directories of the single-model runs.
P="${MATPES_RUNS:?directory containing the per-model *_paper directories}"

time srun ./eval_with_report_efs.py \
    --aggregate \
        "$P/v14_paper" \
        "$P/omat0_paper" \
        "$P/direct_paper" \
        "$P/rdm1_paper" \
        "$P/rdm2_paper" \
        "$P/rdm3_paper" \
        "$P/rdm4_paper" \
    --labels v14 OMAT0 DIRECT rdm1 rdm2 rdm3 rdm4 \
    --ceiling "$P/omat0_paper" \
    --aggregate_out matpes_paper_tables
