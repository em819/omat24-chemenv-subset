#!/bin/bash -l
#SBATCH --job-name=litraj-nebdft2k
#SBATCH --account=CHANGE_ME            # was p200243                 # <-- your project id
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=0-06:00:00                 # 1656 hops (median 111 atoms; 5 images 73% / 7 images 27%, mean 3.55 interior). FIRE band capped at 100 steps, so <=355 band force calls + endpoints; assume ~500/hop. 1656*500/16 ranks = 52k calls per rank; at 50 ms that is ~43 min, at 150 ms ~2.2 h. The tail, not the median, sets this: the notebook's source relaxation is UNCAPPED, and (retries+1)*per-run-timeout = 40 min per stuck hop. SMOKE-TEST FIRST (see below).
#SBATCH --nodes=4                         # 4 GPUs per node -> 16 ranks
#SBATCH --ntasks-per-node=4               # one task per A100 on the node
#SBATCH --gres=gpu:4                      # whole node: 4x A100-40GB
#SBATCH --cpus-per-task=32                # 128 cores / 4 tasks
#SBATCH --output=litraj-nebdft2k-%j.out
#SBATCH --error=litraj-nebdft2k-%j.err
#SBATCH --requeue                         # .done markers let requeued jobs resume

# WALLTIME BASIS: step counts MEASURED from the authors' published MACE-MP-0 runs with this
# exact protocol (their neb.traj files hold n_images frames per optimiser step -- verified
# 100% divisible). ASE evaluates only the interior images each step, so force calls per
# item = steps*n_interior + endpoint relaxations. The one unmeasured factor is the per-call
# cost of YOUR checkpoint; the --limit smoke test gives it, then rescale linearly.

set -euo pipefail

# >>> EDIT <<< must match the MLIP Arena env; this benchmark reuses it
VENV="${ARENA_VENV:?set ARENA_VENV -- see envs/envD-mliparena}"
# <-- your checkpoint (the ONLY thing you normally change between runs)
MODEL="/project/scratch/p200243/evgeny/Projects/OMat24_for_So3lr/active_learning_of_OMat24/4th_generation_tokens_train-mlff/mace-models/v14_T100_alpha1_support_material/be1024_universal_loss_dft_e0s_use_stress_delta_0.01_lr_0.001_num_radial_basis10/compile_model/converged/mace-subset-2nd_gen_chemenv_v14_T100_a1_suppm-omat-medium-be1024_universal_loss_dft_e0s_use_stress_delta_0.01_lr_0.001_rb10_run-1.model"   # <-- your checkpoint
MODEL_NAME="mace-omat-medium-subset-v14_T100_a1_suppm_0.01"

OUTDIR="$SLURM_SUBMIT_DIR/results_litraj_nebdft2k"
CACHE_DIR="/project/scratch/p200243/evgeny/Projects/OMat24_for_So3lr/active_learning_of_OMat24/2nd_generation_tokens_train-mlff/benchmarks/neb_extended/reference_structures_and_values/nebdft2k"     # shared with submit_meluxina_litraj_mplitrj.sh

module load env/release/2024.1            # must match your stability/NEB env exactly
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source "$VENV/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# nebDFT2k is a 65 MB download from the LiTraj mirror. Compute nodes are offline, so it must
# be staged on a LOGIN node first. Fail fast if it is missing.
#   "$VENV/bin/python" run_litraj_nebdft2k.py --prefetch --cache-dir litraj_cache
if [ ! -f "$CACHE_DIR/nebDFT2k/nebDFT2k_index.csv" ]; then
    echo "ERROR: $CACHE_DIR missing nebDFT2k. On a login node run:" >&2
    echo "  \"$VENV/bin/python\" \"$SLURM_SUBMIT_DIR/run_litraj_nebdft2k.py\" --prefetch --cache-dir \"$CACHE_DIR\"" >&2
    exit 1
fi

# --- SMOKE-TEST FIRST. Same reasoning as submit_meluxina_neb.sh: float64 + --cueq is a
# combination neither LiTraj nor your stability run exercised. If MACECalculator rejects it,
# every item crashes and burns the allocation. One short job settles it AND gives you the
# per-hop cost you need to sanity-check the walltime above:
#   ... run_litraj_nebdft2k.py --model "$MODEL" --model-name calib --outdir results_calib \
#       --cache-dir "$CACHE_DIR" --dtype float64 --cueq --limit 8
# If cueq fails there, drop --cueq below: LiTraj ran without it, so that is the more
# faithful configuration anyway.
#
# NOTE ON --tm-policy: left at the default `flag`, which runs all 1656 hops and reports the
# two splits separately at aggregate time. The w/o-Hubbard-U split (646 hops) is the one
# where the model's PBE+U training and nebDFT2k's plain-PBE reference actually agree -- quote
# that as the headline and the with-U split as a diagnostic. Use `--tm-policy drop` only if
# you want to spend the GPU hours solely on the clean subset.
srun --kill-on-bad-exit=0 "$VENV/bin/python" "$SLURM_SUBMIT_DIR/run_litraj_nebdft2k.py" \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --outdir "$OUTDIR" \
    --cache-dir "$CACHE_DIR" \
    --dtype float64 \
    --cueq \
    --element-policy supported \
    --tm-policy flag \
    --symmetric-endpoints \
    --endpoint-fmax 0.05 \
    --endpoint-steps 10000 \
    --band-fmax 0.05 \
    --n-steps 10000 \
    --per-run-timeout 3600 || true

# --- Aggregation: single process (NOT srun) merges the per-hop JSONs into parquet and prints
# MAE / RMSE / Rp / slope / R2 via litraj.metrics.get_metrics, split by Hubbard-U chemistry,
# under both the published converged-only gate and the all-items view.
echo "NEB step finished. Aggregating..."
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/run_litraj_nebdft2k.py" \
    --aggregate \
    --model-name "$MODEL_NAME" \
    --outdir "$OUTDIR" \
    --cache-dir "$CACHE_DIR" \
    --n-steps 10000

echo "Done."
echo "  $OUTDIR/${MODEL_NAME}-litraj-nebdft2k.parquet   (per-hop E_m, PA-MAE, convergence)"
echo "  $OUTDIR/bands/                                  (relaxed bands, extxyz)"
# NB: with --element-policy supported the 25 hops containing Eu or La are removed from the
# work-list up front and listed in dropped_structures.csv.
