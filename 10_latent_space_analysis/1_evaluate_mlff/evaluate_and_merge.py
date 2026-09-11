#!/usr/bin/env python3
"""Stage 10.1 -- run the model over a reference set and merge its predictions
with the DFT reference into one extxyz file.

Writes <outdir>/all_structures_ref_and_predicted.xyz, holding the predicted
energy and forces alongside REF_energy and REF_forces. Stage 10.2 reads it.

Environment: envB-mace. One GPU, about 2 h for the 657,756-structure set.
"""

import argparse
import os
import sys

import ase.io
from mace.cli.eval_configs import main as mace_eval_configs_main

DEFAULT_REF = (
    "/project/home/p200715/evgeny/Projects/OMat24_for_So3lr/construct_minimal_omat_subset/"
    "train_mlff/mace-work/prepare_train_val_and_test_set/test_set/no_rare_earths/"
    "all_structures.xyz"
)
DEFAULT_MODEL = (
    "/project/scratch/p200243/evgeny/Projects/OMat24_for_So3lr/active_learning_of_OMat24/"
    "train-mlff/mace-models/mace-omat24-medium/mace-omat-0-medium.model"
)


def run_eval(configs, model, output, device, dtype):
    sys.argv = [
        "program",
        "--configs", configs,
        "--model", model,
        "--output", output,
        "--device", device,
        "--default_dtype", dtype,
    ]
    mace_eval_configs_main()


def merge(ref_xyz, pred_xyz, out_xyz):
    print("Loading reference xyz", flush=True)
    ref_atoms = ase.io.read(ref_xyz, index=":")
    print("Loading predicted xyz", flush=True)
    pred_atoms = ase.io.read(pred_xyz, index=":")

    if len(ref_atoms) != len(pred_atoms):
        raise SystemExit(
            f"frame count mismatch: reference {len(ref_atoms)} vs prediction "
            f"{len(pred_atoms)}. The merge is positional -- refusing to guess."
        )

    print("Merging xyzs", flush=True)
    for i, (atoms_in, atoms_out) in enumerate(zip(ref_atoms, pred_atoms)):
        if i % 10000 == 0:
            print(f"Processing structure No {i}", flush=True)
        atoms_out.info["REF_energy"] = atoms_in.get_potential_energy()
        atoms_out.arrays["REF_forces"] = atoms_in.get_forces()

    ase.io.write(out_xyz, pred_atoms, write_info=True, write_results=True)
    print(f"wrote {out_xyz} ({len(pred_atoms)} structures)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default=DEFAULT_REF,
                   help="reference extxyz with DFT energies/forces")
    p.add_argument("--model", default=DEFAULT_MODEL, help="MACE .model file")
    p.add_argument("--outdir", default=".",
                   help="where the prediction and merged files are written")
    p.add_argument("--pred-name", default="predictions.xyz",
                   help="intermediate per-frame prediction file")
    p.add_argument("--out-name", default="all_structures_ref_and_predicted.xyz",
                   help="merged output consumed by stage 10.2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--skip-eval", action="store_true",
                   help="predictions already exist; only redo the merge")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pred_path = os.path.join(args.outdir, args.pred_name)
    out_path = os.path.join(args.outdir, args.out_name)

    if not args.skip_eval:
        run_eval(args.configs, args.model, pred_path, args.device, args.dtype)

    merge(args.configs, pred_path, out_path)


if __name__ == "__main__":
    main()
