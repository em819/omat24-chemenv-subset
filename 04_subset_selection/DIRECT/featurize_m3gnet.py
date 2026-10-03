#!/usr/bin/env python3
# As-run filename: featurize_m3gnet_v2.py
"""DIRECT step 2: featurise structures with the M3GNet formation-energy model.

Qi et al., npj Comput. Mater. 10, 43 (2024) use the 128-dimensional output of
M3GNet's final graph-convolution layer. This reproduces it with the matgl port
("M3GNet-Eform-MP-2018.6.1", the default encoder of the authors' `maml`
package): the "readout" feature of model.predict_structure(return_features=True),
exactly as maml.describers.MatGLStructure computes it.

matgl >= 3.0.5 does not load the 12 Set2Set readout LSTM tensors of this
checkpoint (renamed keys, loaded with strict=False) and leaves them randomly
initialised. _restore_checkpoint_weights loads them and checks that the model
equals the checkpoint, so the features equal those of maml with matgl 3.0.3,
which loads the checkpoint completely.

Output: an (N, 128) float32 array <out>.npy and <out>.ids.json, which maps each
row to "<file basename>:<frame index>".

Inputs (--input): an ase.io-readable file, a glob, a directory of
.extxyz/.xyz/.cif/.vasp/.POSCAR files, an ASE database (.db / .aselmdb), or a
.json list of pymatgen Structure dicts.

Sharding (--shard i/N):
  stride  row idx goes to shard idx % N (any input; every shard parses all rows)
  file    shard i reads only files[i::N] of the sorted file list (many-file
          inputs, e.g. the chunks of split_xyz_shards.py). Used for the paper.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def _iter_structures(input_path: str, limit: int | None = None,
                     shard_i: int | None = None, shard_n: int | None = None,
                     shard_by: str = "stride"):
    """Yield (identifier, pymatgen.Structure) pairs for this shard."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    count = 0

    def _take(idx):
        return shard_n is None or (idx % shard_n) == shard_i

    if input_path.endswith((".db", ".aselmdb")):
        from ase.db import connect

        db = connect(input_path)
        idx = 0
        for row in db.select():
            if _take(idx):
                if limit is not None and count >= limit:
                    return
                atoms = row.toatoms()
                ident = str(getattr(row, "id", idx))
                yield ident, adaptor.get_structure(atoms)
                count += 1
            idx += 1
        return

    if input_path.endswith(".json"):
        with open(input_path) as fh:
            data = json.load(fh)
        for idx, d in enumerate(data):
            if not _take(idx):
                continue
            if limit is not None and count >= limit:
                return
            yield str(idx), Structure.from_dict(d)
            count += 1
        return

    if os.path.isdir(input_path):
        files = sorted(
            f for f in glob.glob(os.path.join(input_path, "*"))
            if f.endswith((".extxyz", ".xyz", ".cif", ".vasp", ".POSCAR"))
        )
    else:
        files = sorted(glob.glob(input_path)) or [input_path]

    from ase.io import iread

    take_all = False
    if shard_by == "file" and shard_n is not None:
        files = files[shard_i::shard_n]
        take_all = True

    idx = 0
    for f in files:
        for j, atoms in enumerate(iread(f)):
            if take_all or _take(idx):
                if limit is not None and count >= limit:
                    return
                ident = f"{os.path.basename(f)}:{j}"
                yield ident, adaptor.get_structure(atoms)
                count += 1
            idx += 1


DEFAULT_MODEL = "M3GNet-Eform-MP-2018.6.1"  # maml.describers._matgl.DEFAULT_MODEL

# The checkpoint names the readout LSTM "readout.set2set.lstm.*"; matgl >= 3.0.5
# expects "readout.lstm.*".
_KEY_RENAMES = (("readout.set2set.lstm.", "readout.lstm."),)
# Buffers the model builds from model.json and that the checkpoint does not store.
_BUFFERS_NOT_IN_CHECKPOINT = ("basis_expansion.sbf.cutoff", "bond_expansion.rbf.cutoff")


def _restore_checkpoint_weights(module, name: str, prefix: str = ""):
    """Load every tensor of the named checkpoint into `module`, renaming keys the
    installed matgl does not recognise, and verify the result.

    `module` is what matgl.load_model(name) returns (prefix="") or its .model
    (prefix="model.", the part that maml.describers.MatGLStructure keeps).
    Raises RuntimeError if a checkpoint tensor has no place in the model, if a
    tensor differs after loading, or if a model tensor is not in the checkpoint
    (other than the cutoff buffers).
    """
    from pathlib import Path

    import torch
    from matgl.utils.io import _get_file_paths  # the resolver load_model itself uses

    fpaths = _get_file_paths(Path(name), str_path=str(name))
    state = torch.load(fpaths["state.pt"], map_location="cpu", weights_only=False)

    target = {k: v for k, v in module.state_dict().items()}
    ckpt = {}
    for key, tensor in state.items():
        if not key.startswith(prefix):
            raise RuntimeError(f"checkpoint key {key!r} lacks the expected prefix {prefix!r}")
        key = key[len(prefix):]
        if key not in target:
            for old, new in _KEY_RENAMES:
                key = key.replace(old, new)
        ckpt[key] = tensor

    unplaced = sorted(k for k in ckpt if k not in target)
    if unplaced:
        raise RuntimeError(f"checkpoint tensors with no counterpart in the model: {unplaced}")
    shape_bad = sorted(k for k in ckpt if ckpt[k].shape != target[k].shape)
    if shape_bad:
        raise RuntimeError(f"checkpoint/model shape mismatch: {shape_bad}")

    renamed = sorted(k for k in ckpt if not torch.equal(ckpt[k], target[k].detach().cpu()))
    if renamed:
        module.load_state_dict({k: ckpt[k] for k in renamed}, strict=False)

    after = module.state_dict()
    still_bad = sorted(k for k in ckpt if not torch.equal(ckpt[k], after[k].detach().cpu()))
    uncovered = sorted(k for k in after if k not in ckpt
                       and not any(k.endswith(b) for b in _BUFFERS_NOT_IN_CHECKPOINT))
    if still_bad or uncovered:
        raise RuntimeError(f"model weights do not match the checkpoint: differ={still_bad} "
                           f"not in checkpoint={uncovered}")
    print(f"[encoder] checkpoint verified: {len(ckpt)} tensors; "
          f"{len(renamed)} restored that matgl had not loaded {renamed}", file=sys.stderr)
    return renamed


def _load_encoder(model_name, device: str, prefer_maml: bool):
    """Return a callable Structure -> 1-D feature vector.

    Uses maml.describers.MatGLStructure if maml is installed, otherwise calls
    matgl directly in the same way. (maml was not installed for the paper's
    run, so the direct matgl path produced the published features.)
    """
    if prefer_maml:
        try:
            from maml.describers import MatGLStructure

            enc = MatGLStructure(model_path=model_name)
            _restore_checkpoint_weights(enc.describer_model, enc.model_path, prefix="model.")

            def featurize(struct):
                return np.asarray(enc.transform_one(struct))

            print(f"[encoder] maml.describers.MatGLStructure(model_path={model_name})", file=sys.stderr)
            return featurize
        except Exception as exc:  # noqa: BLE001
            print(f"[encoder] maml unavailable ({exc}); falling back to direct matgl", file=sys.stderr)

    import matgl
    import torch

    matgl.set_backend("DGL")
    name = model_name or DEFAULT_MODEL
    loaded = matgl.load_model(name)
    _restore_checkpoint_weights(loaded, name)
    model = getattr(loaded, "model", loaded)
    model = model.to(device)

    def featurize(struct):
        # No torch.no_grad(): maml has none, and with it the features differ by
        # 1-2 float32 ulp.
        results = model.predict_structure(struct, return_features=True)
        return np.array(torch.squeeze(results["readout"]).detach().cpu().numpy())

    print(f"[encoder] matgl.load_model('{name}').model on {device}", file=sys.stderr)
    return featurize


def main():
    ap = argparse.ArgumentParser(description="DIRECT step 2: M3GNet featurization.")
    ap.add_argument("--input", required=True, help="structures: file / glob / dir / .db / .aselmdb / .json")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.npy and <out>.ids.json")
    ap.add_argument("--model", default=None, help=f"matgl model name (default {DEFAULT_MODEL}).")
    ap.add_argument("--device", default="cpu",
                    help="cpu (as in the paper) | cuda (faster; float differences can move "
                         "borderline cluster assignments).")
    ap.add_argument("--limit", type=int, default=None, help="cap on structures per shard (benchmarking).")
    ap.add_argument("--checkpoint-every", type=int, default=5000,
                    help="flush features to disk every N structures.")
    ap.add_argument("--shard", default=None, help="'i/N': process only shard i (0-indexed) of N.")
    ap.add_argument("--shard-by", choices=["stride", "file"], default="stride")
    ap.add_argument("--no-maml", action="store_true", help="call matgl directly even if maml is installed.")
    args = ap.parse_args()

    shard_i = shard_n = None
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    device = args.device
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, using CPU", file=sys.stderr)
            device = "cpu"
    except Exception:
        device = "cpu"

    featurize = _load_encoder(args.model, device, prefer_maml=not args.no_maml)

    feats: list[np.ndarray] = []
    ids: list[str] = []
    out_npy = args.out + ".npy"
    out_ids = args.out + ".ids.json"

    def flush():
        arr = np.asarray(feats, dtype=np.float32)
        np.save(out_npy, arr)
        with open(out_ids, "w") as fh:
            json.dump(ids, fh)

    if args.shard_by == "file" and args.input.endswith((".db", ".aselmdb", ".json")):
        print("[warn] --shard-by file has no effect on a single .aselmdb/.json input; using stride.",
              file=sys.stderr)

    for ident, struct in _iter_structures(args.input, args.limit, shard_i, shard_n, args.shard_by):
        feats.append(featurize(struct))
        ids.append(ident)
        if len(feats) % args.checkpoint_every == 0:
            flush()
            print(f"[featurize] {len(feats)} structures done (dim={len(feats[0])})", file=sys.stderr)

    if not feats:
        if args.shard_by == "file" and shard_n is not None:
            # More shards than files: this shard has nothing to do.
            print(f"[info] shard {shard_i}/{shard_n} (file mode) had no files assigned; "
                  f"nothing to do.", file=sys.stderr)
            return
        print("[error] no structures were featurized; check --input", file=sys.stderr)
        sys.exit(1)

    flush()
    arr = np.asarray(feats, dtype=np.float32)
    print(f"[done] wrote {arr.shape[0]} x {arr.shape[1]} features -> {out_npy}", file=sys.stderr)


if __name__ == "__main__":
    main()
