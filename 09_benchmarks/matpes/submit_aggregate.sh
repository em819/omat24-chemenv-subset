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

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MACE_VENV     REQUIRED. Path to the env B virtualenv DIRECTORY -- see
#               envs/envB-mace/.
#
# MATPES_RUNS   REQUIRED. The parent DIRECTORY holding the `*_paper`
#               directories that submit_eval_single.sh produced -- one per
#               model. Each RUN_DIRS entry below is resolved inside it.
#
# RUN_DIRS      The `*_paper` directories to aggregate, as a bash ARRAY, given
#               relative to $MATPES_RUNS (absolute paths also work).
# LABELS        One label per RUN_DIRS entry, SAME ORDER, same count.
#               >>> Any label starting with rdm / seed / random is treated as a
#               random-seed REPLICA defining the noise floor. <<<
# CEILING_DIR   The `*_paper` directory of the CEILING model (the paper used
#               MACE-OMAT-0). Pointing it at a directory that is also in
#               RUN_DIRS is deliberate: that model's own excess_median then
#               comes out as exactly 0.000 in every domain, which is a
#               self-check on the ceiling arithmetic.
# OUT_DIR       Where the aggregated tables are written.
#
# NON_BULK_DETECTOR
#               Passed as --aggregate_non_bulk_detector. It recomputes the
#               bulk / non-bulk split from the geometric descriptors already
#               stored in every per_structure.npz, instead of using the split
#               saved at evaluation time, and applies it identically to every
#               model AND to the ceiling. The paper's tables use 'dim'
#               (periodic dimensionality < 3), so leave it as it is to
#               reproduce paper_tables/. Set it to the empty string to use the
#               classification each evaluation stored.
#
# THE DEFAULTS BELOW ARE THE PAPER'S SEVEN MODELS and are a TEMPLATE: the
# EnvTok model (v14), MACE-OMAT-0 (public), the DIRECT model and four random-seed
# replicas (trained from the subsets of 04_subset_selection/DIRECT and
# 04_subset_selection/random with stages 06-08). Cut RUN_DIRS/LABELS down to the
# models you actually have. With no rdm* entry there is no noise floor.
#
#     MATPES_RUNS=/path/to/runs sbatch --account=<your allocation> submit_aggregate.sh
# ============================================================================
MACE_VENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
P="${MATPES_RUNS:?set MATPES_RUNS to the directory containing the per-model *_paper directories}"
RUN_DIRS=(v14_paper omat0_paper direct_paper rdm1_paper rdm2_paper rdm3_paper rdm4_paper)
LABELS=(v14 OMAT0 DIRECT rdm1 rdm2 rdm3 rdm4)
CEILING_DIR="omat0_paper"
NON_BULK_DETECTOR="dim"
OUT_DIR="matpes_paper_tables"
# ============================================================================

if [ "${#RUN_DIRS[@]}" -ne "${#LABELS[@]}" ]; then
    echo "ERROR: RUN_DIRS has ${#RUN_DIRS[@]} entries but LABELS has ${#LABELS[@]}." >&2
    echo "       They are positional -- give exactly one label per directory." >&2
    exit 1
fi

# Resolve each entry against $P unless it is already absolute.
resolve() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$P" "$1" ;; esac; }

AGG_DIRS=()
for d in "${RUN_DIRS[@]}"; do
    full="$(resolve "$d")"
    if [ ! -d "$full" ]; then
        echo "ERROR: $full does not exist. Each RUN_DIRS entry must be a" >&2
        echo "       *_paper directory written by submit_eval_single.sh." >&2
        exit 1
    fi
    AGG_DIRS+=("$full")
done

module load env/release/2024.1
module load CUDA/12.6.0

conda deactivate 2>/dev/null || true
source "$MACE_VENV/bin/activate"

time srun ./eval_with_report_efs.py \
    --aggregate "${AGG_DIRS[@]}" \
    --labels "${LABELS[@]}" \
    --ceiling "$(resolve "$CEILING_DIR")" \
    ${NON_BULK_DETECTOR:+--aggregate_non_bulk_detector "$NON_BULK_DETECTOR"} \
    --aggregate_out "$OUT_DIR"
