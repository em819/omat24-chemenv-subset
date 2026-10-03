#!/usr/bin/env python3
# As-run filename: extract_omat24_subset.py
"""
Stage 00: convert one OMat24 sub-dataset from its ASE databases (.aselmdb) to
one extended-xyz file, dropping structures that contain a rare earth or an
actinide.

Each frame is written with ase.io.write(format="extxyz", write_info=True,
write_results=True), so the comment line carries sid= and the other OMat24
metadata, the energy and the stress, and the atom lines carry the species,
positions, initial_magmoms (where present) and forces. One key is added:
multiplicity = sum over atoms of |round(initial magnetic moment)|.

MPI: rank r of N converts the contiguous block of structure indices
[r*n/N, (r+1)*n/N) into <output>.rank<r>, and rank 0 joins the rank files in
rank order. The output therefore follows the database order and does not
depend on the number of ranks.

USAGE
  mpirun -n 128 python convert_omat24.py \\
      --db-dir  omat24/train/rattled-relax \\
      --output  omat24_xyz/rattled-relax/all_structures_rattled-relax.xyz
"""

import argparse
import os
import time

import ase.io
import numpy as np
from fairchem.core.datasets import AseDBDataset
from mpi4py import MPI

RARE_EARTHS = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
               'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
               'Fm', 'Md', 'No', 'Lr'}


def contains_rare_earth(atoms):
    return bool(set(atoms.get_chemical_symbols()) & RARE_EARTHS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-dir", required=True,
                    help="directory holding the .aselmdb files of ONE sub-dataset.")
    ap.add_argument("--output", required=True, help="output extxyz file.")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    dataset = AseDBDataset(config=dict(src=args.db_dir,
                                       a2g_args={"r_energy": True, "r_forces": True, "r_stress": True}))
    n = dataset.num_samples
    start, stop = rank * n // size, (rank + 1) * n // size
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        print(f"{args.db_dir}: {n} structures on {size} ranks", flush=True)
    comm.barrier()

    t0 = time.time()
    rank_file = f"{args.output}.rank{rank}"
    n_written = n_skipped = 0
    with open(rank_file, "w") as fh:
        for idx in range(start, stop):
            atoms = dataset.get_atoms(idx)
            if contains_rare_earth(atoms):
                n_skipped += 1
                continue
            atoms.info['multiplicity'] = int(np.sum(np.abs(np.rint(atoms.get_initial_magnetic_moments()))))
            ase.io.write(fh, atoms, format='extxyz', write_info=True, write_results=True, parallel=False)
            n_written += 1

    n_written = comm.reduce(n_written, root=0)
    n_skipped = comm.reduce(n_skipped, root=0)
    comm.barrier()

    if rank == 0:
        with open(args.output, "w") as out:
            for r in range(size):
                part = f"{args.output}.rank{r}"
                with open(part) as fh:
                    while True:
                        block = fh.read(1 << 24)
                        if not block:
                            break
                        out.write(block)
                os.remove(part)
        print(f"wrote {n_written} structures, skipped {n_skipped} with rare earths "
              f"-> {args.output}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
