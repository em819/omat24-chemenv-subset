#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4          # single-threaded + I/O-bound; cores don't speed it up
#SBATCH --time=12:00:00            # one streaming pass over the source xyz; bump to 1-00:00:00 if huge
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J gen-train-set
#SBATCH --output=gen-train-set-%j.out
#SBATCH --error=gen-train-set-%j.err

# Materialise ONE training set: read a file of OMat24 mat_ids and copy the
# matching structures out of the source extxyz into a single train.xyz.
# (Validation = the official OMat24 valid split, so no split is done here.)
#
# Unlike the k-fold partition job this is light: no DuckDB, no 128-core greedy,
# no big RAM matrix -- just numpy (to read the id list) + sequential file I/O.
# The cost is one full scan of the source xyz regardless of subset size, so the
# wall time is set by the TOTAL source data, not the subset.

deactivate 2>/dev/null || true
conda activate active_learning_omat24_python310

# Keep BLAS single-threaded (genfromtxt is the only numpy use; harmless anyway).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- EDIT THESE PER SUBSET --------------------------------------------------
SUBSET="/project/scratch/p200243/evgeny/Projects/OMat24_for_So3lr/active_learning_of_OMat24/subset_construction/14_per_atom_score_with_dists_and_A_deep/T100_alpha1_support_material/best_subset_coarsened_v14andAdeep_T100_R0full_asymfloor_designa_supmaterials_ncov1_alpha1_uw0_topup.txt"
XYZDIR="/project/home/p200243/tcp_datasets/OMat24_data/original_train_valid_split/train_ready_for_so3lr/all_subsets"
OUTPUT="subset_v14_T100_alpha1_support_material_train.xyz"
# -----------------------------------------------------------------------------

python -u ./generate_train_set.py \
    --subset-mat-ids "$SUBSET" \
    --xyz-dir        "$XYZDIR" \
    --output         "$OUTPUT" \
    > "out_${OUTPUT%.xyz}.log" 2>&1

echo "Done. Check the tail of out_${OUTPUT%.xyz}.log: 'mat_ids matched X / Y' should be ~100%."
