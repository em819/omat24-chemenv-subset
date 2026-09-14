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

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --configs  REQUIRED. The reference extxyz FILE carrying DFT energies and
#              forces, which the model is evaluated on. The published run used
#              the SAME file stage 06 uses as its valid_file: the official
#              OMat24 validation split converted to extxyz with rare earths
#              removed (657,756 structures / 12,019,558 atoms). Everything
#              downstream in stage 10 is sized by this file. See DATA.md.
#
#   --model    REQUIRED. The MACE .model FILE whose predictions are analysed.
#              The published latent-space analysis used MACE-OMAT-0 medium
#              (mace-omat-0-medium.model, a public foundation model) -- the same
#              file stage 10.3 extracts features with. Pass your own
#              *_run-1.model from stage 08 to analyse the subset model instead;
#              use the SAME model here and in stage 10.3.
#
#   --outdir   Where predictions.xyz and the merged
#              all_structures_ref_and_predicted.xyz are written. Stage 10.2
#              reads the merged file as ../all_structures_ref_and_predicted.xyz,
#              so point --outdir at the PARENT of the directory you will run
#              stage 10.2 from.
#
# Neither --configs nor --model has a default: both used to be absolute paths
# on the author's cluster, which silently pointed everyone else at files they
# do not have.
# ============================================================================


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
    p.add_argument("--configs", required=True,
                   help="REQUIRED: reference extxyz with DFT energies/forces "
                        "(the published run used the OMat24 validation split, "
                        "rare earths removed -- see DATA.md)")
    p.add_argument("--model", required=True,
                   help="REQUIRED: MACE .model file to evaluate (the published "
                        "run used mace-omat-0-medium.model; use the same model "
                        "here and in stage 10.3)")
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
