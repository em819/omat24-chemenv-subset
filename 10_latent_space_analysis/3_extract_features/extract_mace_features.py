#!/usr/bin/env python3
"""
Stage 1 of the chemenv-feature-space study: extract per-atom MACE invariant
feature vectors for every atom of the evaluated test set.

WHAT IS THE FEATURE VECTOR?  (verified against the MACE codebase)
    MACECalculator.get_descriptors(atoms, invariants_only=True) returns the
    model's per-atom `node_feats` -- the concatenation, over all interaction
    layers, of the l=0 (invariant) part of each layer's post-product node
    features. In mace/modules/models.py the ENERGY READOUTS consume exactly
    these node_feats (node_es = readout(node_feats_concat[i], ...)), and
    node_feats_out is built BEFORE the force autograd pass. So these invariants
    are precisely the latent representation the model's own predictions are
    built from -- the correct "how MACE sees this atom" vector. (Confirmed in
    mace/calculators/mace.py:get_descriptors and mace/modules/models.py:forward.)

WHY NOT JUST LOOP calculator.get_descriptors() PER STRUCTURE?
    get_descriptors() calls model(batch) with the default compute_force=True, so
    every call runs a full autograd backward for forces we then discard, and it
    processes ONE structure at a time. This script keeps the calculator only to
    LOAD the model (it handles dtype / device / z_table / r_max / heads / optional
    cueq), then bypasses get_descriptors with a faster path:
        * batches many structures per forward (GPU utilisation),
        * model(..., compute_force=False, compute_stress=False) under no_grad
          (skips the force backward entirely),
        * parallel CPU-side neighbour-list / graph construction via DataLoader
          workers so the GPU is not starved.
    The invariant slicing replicates mace.modules.utils.extract_invariant exactly,
    and a startup self-check asserts the batched features match
    calculator.get_descriptors() on the first structure to float epsilon.

PARALLELISM ON THE HPC
    The work is GPU-bound and embarrassingly parallel over structures. Run one
    process per GPU as a SLURM array job; pass --shard $SLURM_ARRAY_TASK_ID and
    --nshards $SLURM_ARRAY_TASK_COUNT. Each shard processes structures[shard::N],
    writes its own features_shard{ID}.npy (float32 memmap) + meta_shard{ID}.parquet.
    Stage 2 (chemenv_feature_metrics.py) globs all shards back together.

    Example sbatch (4 GPUs -> 4 array tasks):
        #SBATCH --array=0-3
        #SBATCH --gres=gpu:1
        srun python extract_mace_features.py \
            --xyz structures_with_errors_and_v2chemenvs_....xyz \
            --model /path/to/mace-omat-0-small.model \
            --out-dir mace_features \
            --shard $SLURM_ARRAY_TASK_ID --nshards $SLURM_ARRAY_TASK_COUNT \
            --batch-size 64 --num-workers 8

RUN ENVIRONMENT: this script must run in your HPC MACE environment (the same
one that runs analyze_prediction_v2_..._opt.py), NOT the pymatgen env -- it needs
torch + mace + e3nn + torch_geometric. Stage 2 needs none of those.
"""

import argparse
import os
import time
from io import StringIO

import numpy as np
import pandas as pd

import ase.io
import torch

# REPO CHANGE: mace must be imported BEFORE e3nn. `mace/__init__.py` sets
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, and e3nn does a bare torch.load of its
# Wigner constants at import time. From torch 2.6 onward torch.load defaults to
# weights_only=True, so importing e3nn first fails with
#   UnpicklingError: Weights only load failed ... Unsupported global: GLOBAL slice
# env B pins torch 2.5.0, where the old default hides this; the ordering makes
# the script work on newer torch too.
from mace.calculators import MACECalculator
from mace import data as mace_data
from mace.tools import torch_geometric
from mace.modules.utils import extract_invariant

from e3nn import o3


# ----------------------------------------------------------------------------
# Random-access extxyz reader (byte-offset index), adapted from
# analyze_prediction_v2_..._opt.py. ONE sequential pass records each frame's
# byte offset AND atom count, then we seek directly to the frames this shard
# owns. The atom counts let us preallocate the output memmap exactly.
# ----------------------------------------------------------------------------
def build_frame_index(path):
    """Return (offsets, natoms) lists, one entry per frame."""
    offsets, natoms = [], []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                nat = int(s)
            except ValueError:
                raise RuntimeError(
                    f"{path}: expected integer natoms at byte {pos}, got {s[:60]!r}")
            offsets.append(pos)
            natoms.append(nat)
            f.readline()                 # comment / Lattice line
            for _ in range(nat):
                f.readline()             # atom lines
    return offsets, natoms


def read_frame_at(path, offset):
    """Seek to `offset` and parse exactly one extxyz frame into ase.Atoms."""
    with open(path, "rb") as f:
        f.seek(offset)
        nat = int(f.readline())
        rest = [f.readline() for _ in range(nat + 1)]
        text = str(nat) + "\n" + b"".join(rest).decode()
    return ase.io.read(StringIO(text), format="extxyz", index=0, parallel=False)


# ----------------------------------------------------------------------------
# A torch Dataset that builds the MACE graph (AtomicData) lazily, so DataLoader
# worker processes do the neighbour-list construction in parallel on CPU while
# the GPU is busy with the previous batch. shuffle=False keeps batch order ==
# dataset order, so we can line features up with metadata by simple counting.
# ----------------------------------------------------------------------------
class _GraphDataset(torch.utils.data.Dataset):
    def __init__(self, atoms_list, z_table, r_max, heads, head):
        self.atoms_list = atoms_list
        self.z_table = z_table
        self.r_max = r_max
        self.heads = heads
        self.head = head

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, i):
        config = mace_data.config_from_atoms(self.atoms_list[i], head_name=self.head)
        return mace_data.AtomicData.from_config(
            config, z_table=self.z_table, cutoff=self.r_max, heads=self.heads)


def invariant_params(model):
    """Replicate the (l_max, num_features, num_layers) used by get_descriptors."""
    num_interactions = int(model.num_interactions)
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_features = irreps_out.dim // (l_max + 1) ** 2
    return l_max, num_features, num_interactions


def forward_invariants(model, batch, l_max, num_features, num_layers):
    """node_feats -> per-atom invariant descriptor (n_atoms_in_batch, D)."""
    with torch.no_grad():
        out = model(batch.to_dict(), compute_force=False, compute_stress=False,
                    training=False)
    inv = extract_invariant(out["node_feats"], num_layers=num_layers,
                            num_features=num_features, l_max=l_max)
    return inv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xyz", required=True,
                    help="extxyz from analyze_prediction (has chemenv_id, f_ae_mace, sid).")
    ap.add_argument("--model", required=True, help="Path to the MACE .model file.")
    ap.add_argument("--out-dir", default="mace_features")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers for parallel graph construction.")
    ap.add_argument("--window", type=int, default=4000,
                    help="Structures held in RAM at once (bounds memory).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--enable-cueq", action="store_true",
                    help="Use cuequivariance acceleration (match your eval setup).")
    ap.add_argument("--max-structures", type=int, default=None,
                    help="Debug: cap structures processed (after sharding).")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the get_descriptors equivalence self-check.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"shard{args.shard:03d}"

    # ---- Load the model via the calculator (handles dtype/device/cueq/z_table) ----
    print(f"[{tag}] loading MACE model ...", flush=True)
    calc = MACECalculator(model_paths=args.model, device=args.device,
                          default_dtype=args.dtype, enable_cueq=args.enable_cueq)
    if calc.num_models != 1:
        raise SystemExit("This prototype expects a single model, not a committee.")
    model = calc.models[0]
    model.eval()
    l_max, num_features, num_layers = invariant_params(model)
    D = num_features * num_layers
    print(f"[{tag}] invariant descriptor dim D = {D} "
          f"(num_features={num_features} x layers={num_layers}, l_max={l_max})",
          flush=True)

    # ---- Index the file and pick this shard's frames ----
    t0 = time.time()
    offsets, natoms = build_frame_index(args.xyz)
    n_total = len(offsets)
    shard_ids = list(range(n_total))[args.shard::args.nshards]
    if args.max_structures is not None:
        shard_ids = shard_ids[:args.max_structures]
    n_atoms_shard = int(sum(natoms[i] for i in shard_ids))
    print(f"[{tag}] indexed {n_total} frames in {time.time()-t0:.1f}s; "
          f"this shard: {len(shard_ids)} structures / {n_atoms_shard} atoms",
          flush=True)
    if not shard_ids:
        open(os.path.join(args.out_dir, f"meta_{tag}.parquet"), "w").close()
        return

    # ---- Preallocate the output memmap exactly ----
    feat_path = os.path.join(args.out_dir, f"features_{tag}.npy")
    feats = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float32,
                                      shape=(n_atoms_shard, D))
    meta_chemenv, meta_err, meta_elem, meta_sid, meta_lidx = [], [], [], [], []
    meta_fmag = []
    row = 0

    # ---- Optional one-shot equivalence check vs calculator.get_descriptors ----
    if not args.no_validate:
        a0 = read_frame_at(args.xyz, offsets[shard_ids[0]])
        ref = calc.get_descriptors(a0, invariants_only=True)        # (n, D)
        ds = _GraphDataset([a0], calc.z_table, calc.r_max, calc.available_heads, calc.head)
        b0 = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(args.device)
        mine = forward_invariants(model, b0, l_max, num_features, num_layers).cpu().numpy()
        if not np.allclose(ref, mine, atol=1e-4, rtol=1e-3):
            raise SystemExit(f"[{tag}] FEATURE MISMATCH vs get_descriptors "
                             f"(max abs diff {np.abs(ref-mine).max():.2e}) -- aborting.")
        print(f"[{tag}] self-check passed: batched features == get_descriptors "
              f"(max abs diff {np.abs(ref-mine).max():.2e})", flush=True)

    # ---- Windowed streaming over the shard ----
    t0 = time.time()
    for w_start in range(0, len(shard_ids), args.window):
        win_ids = shard_ids[w_start:w_start + args.window]
        atoms_list = [read_frame_at(args.xyz, offsets[i]) for i in win_ids]

        loader = torch_geometric.dataloader.DataLoader(
            _GraphDataset(atoms_list, calc.z_table, calc.r_max,
                          calc.available_heads, calc.head),
            batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=(args.device != "cpu"))

        s_ptr = 0  # index into atoms_list, advanced by num_graphs per batch
        for batch in loader:
            batch = batch.to(args.device)
            inv = forward_invariants(model, batch, l_max, num_features, num_layers)
            inv = inv.cpu().numpy().astype(np.float32, copy=False)
            ptr = batch.ptr.cpu().numpy()           # graph boundaries within batch
            ng = int(batch.num_graphs)
            for g in range(ng):
                atoms = atoms_list[s_ptr + g]
                n = ptr[g + 1] - ptr[g]
                feats[row:row + n] = inv[ptr[g]:ptr[g + 1]]
                meta_chemenv.extend(atoms.arrays["chemenv_id"].tolist())
                meta_err.extend(np.asarray(atoms.arrays["f_ae_mace"], float).tolist())
                # |F_ref| in meV/A (REF_forces is eV/A; x1000 to match f_ae_mace units)
                if "REF_forces" in atoms.arrays:
                    fmag = np.linalg.norm(np.asarray(atoms.arrays["REF_forces"], float),
                                          axis=1) * 1000.0
                else:
                    fmag = np.full(n, np.nan)
                meta_fmag.extend(fmag.tolist())
                meta_elem.extend(atoms.get_chemical_symbols())
                sid = atoms.info.get("sid", "NA")
                meta_sid.extend([sid] * n)
                meta_lidx.extend(range(n))
                row += n
            s_ptr += ng
        done = w_start + len(win_ids)
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[{tag}] {done}/{len(shard_ids)} structures, {row} atoms "
              f"({rate:.1f} struct/s)", flush=True)

    feats.flush()
    assert row == n_atoms_shard, f"row count {row} != preallocated {n_atoms_shard}"

    meta = pd.DataFrame({
        "row": np.arange(n_atoms_shard, dtype=np.int64),
        "chemenv_id": meta_chemenv,
        "f_ae_mace": np.asarray(meta_err, dtype=np.float32),
        "ref_fmag": np.asarray(meta_fmag, dtype=np.float32),
        "element": meta_elem,
        "sid": meta_sid,
        "local_idx": np.asarray(meta_lidx, dtype=np.int32),
    })
    meta_path = os.path.join(args.out_dir, f"meta_{tag}.parquet")
    meta.to_parquet(meta_path, index=False)
    print(f"[{tag}] DONE -> {feat_path} ({n_atoms_shard}x{D} float32) + {meta_path} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
