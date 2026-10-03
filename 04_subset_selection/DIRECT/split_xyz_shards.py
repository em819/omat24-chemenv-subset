"""DIRECT step 1: split extended-xyz files into equal-size chunk files.

xyz has no random access, so featurisation workers would otherwise each parse a
whole multi-million-frame file. Splitting once (plain line-based text I/O, no
ASE) lets every featurisation shard read only its own chunks.

A frame is `natoms + 2` lines. Frames are copied verbatim and rolled into a new
chunk every --frames-per-chunk frames; chunks are named <prefix>_NNNNN.xyz.

Frames containing a rare earth or actinide are dropped, with the element list of
stage 01 (contains_rare_earths), because they are outside the model's z-table.
The paper's input files were already filtered, so nothing was dropped there.
"""

from __future__ import annotations

import argparse
import os
import sys

RARE_EARTHS = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
               'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
               'Fm', 'Md', 'No', 'Lr'}


def _species_column(comment):
    """Column of the species in the atom lines, from the Properties= key."""
    for token in comment.split():
        if token.startswith("Properties="):
            fields = token[len("Properties="):].split(":")
            col = 0
            for i in range(0, len(fields) - 2, 3):
                if fields[i] == "species":
                    return col
                col += int(fields[i + 2])
    return 0


def _has_rare_earth(frame):
    col = _species_column(frame[1])
    return any(line.split()[col] in RARE_EARTHS for line in frame[2:])


def _frames(fh):
    """Yield frames (lists of raw lines) from an open extended-xyz file."""
    while True:
        header = fh.readline()
        if header == "":
            return
        if header.strip() == "":
            continue
        try:
            natoms = int(header.split()[0])
        except (ValueError, IndexError):
            raise ValueError(f"expected an atom-count line, got: {header!r}")
        lines = [header, fh.readline()]
        lines.extend(fh.readline() for _ in range(natoms))
        yield lines


def main():
    ap = argparse.ArgumentParser(description="Split extended-xyz files into equal-size chunk files.")
    ap.add_argument("--input", nargs="+", required=True, help="one or more .xyz/.extxyz files.")
    ap.add_argument("--outdir", required=True, help="directory to write chunk files into.")
    ap.add_argument("--frames-per-chunk", type=int, default=50000)
    ap.add_argument("--prefix", default="chunk", help="chunk filename prefix.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fpc = args.frames_per_chunk
    if fpc < 1:
        ap.error("--frames-per-chunk must be >= 1")

    def chunk_path(i):
        return os.path.join(args.outdir, f"{args.prefix}_{i:05d}.xyz")

    chunk_idx = 0
    frames_in_chunk = 0
    total = 0
    dropped = 0
    writer = open(chunk_path(0), "w")
    try:
        for path in args.input:
            n_this = 0
            with open(path) as fh:
                for frame in _frames(fh):
                    if _has_rare_earth(frame):
                        dropped += 1
                        continue
                    if frames_in_chunk >= fpc:
                        writer.close()
                        chunk_idx += 1
                        writer = open(chunk_path(chunk_idx), "w")
                        frames_in_chunk = 0
                    writer.writelines(frame)
                    frames_in_chunk += 1
                    total += 1
                    n_this += 1
            print(f"[split] {path}: {n_this} frames", file=sys.stderr)
    finally:
        writer.close()

    print(f"[split] wrote {total} frames into {chunk_idx + 1} chunks "
          f"({fpc}/chunk) under {args.outdir}/; dropped {dropped} rare-earth frames",
          file=sys.stderr)


if __name__ == "__main__":
    main()
