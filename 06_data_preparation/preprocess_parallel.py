#!/usr/bin/env python3
"""
Parallel, fork-safe drop-in replacement for `mace.cli.preprocess_data`.

WHY THIS EXISTS
---------------
Stock MACE preprocessing has two scaling problems on large OMat24 subsets:

  1. The READ is single-threaded. `ase.io.read` parses the whole train xyz in one
     process (~56 min for 7.6M configs). `num_process` does NOT help it -- the
     split happens *after* the read.
  2. The fork OOMs. Stock loads ALL configs into one parent, imports torch (which
     reserves a large *virtual* address space, especially with CUDA libs loaded),
     then forks `num_process` copies. Under strict overcommit (vm.overcommit_memory=2,
     common on HPC) the kernel charges parent_VSZ per fork, so 128 forks of a
     torch+all-configs parent exceeds CommitLimit -> `OSError: Cannot allocate memory`
     at `p.start()`, even though physical RAM is fine.

This script keeps MACE's exact parsing / HDF5 / statistics code (so the output is
byte-for-byte what `mace_run_train` expects) but changes ONLY the orchestration:

  * CUDA is disabled before any import (CUDA_VISIBLE_DEVICES="") so per-process
    virtual memory stays small -> forking many workers no longer trips the commit
    limit. Preprocessing never uses a GPU anyway.
  * The xyz is split by FRAME into N shards in one fast text pass (pure I/O, no
    parsing), then the shards are parsed + written to HDF5 IN PARALLEL. The parent
    never holds Configuration objects -> the worker pool forks from a lean parent.
  * Statistics reuse MACE's own `pool_compute_stats` over the produced HDF5 dir
    (the dataset is read lazily from disk, so each worker holds only one batch).

OUTPUT (identical layout to stock preprocess_data):
  <h5_prefix>train/train_*.h5
  <h5_prefix>val/val_*.h5        (only if --valid-file given)
  <h5_prefix>test/test_*.h5      (only if --test-file given)
  <h5_prefix>statistics.json     (only if a train set is processed, unless --no-stats)

USAGE
  # Point it at the SAME yaml you already use; sharding/procs have sane defaults.
  python preprocess_parallel.py --config preprocess.yaml

  # Or all on the CLI:
  python preprocess_parallel.py \
      --train-file train.xyz --valid-file valid.xyz --test-file test.xyz \
      --h5-prefix processed_data/ --r-max 6.0 \
      --energy-key energy --forces-key forces --stress-key stress \
      --atomic-numbers '[1,2,8,...]' --E0s '{1: -1.1, ...}' \
      --read-procs 32 --stats-procs 32 --train-shards 64

  # Preprocess the SHARED valid/test sets ONCE (no train -> no stats), reuse forever:
  python preprocess_parallel.py --valid-file valid.xyz --test-file test.xyz \
      --h5-prefix shared_valtest/ --atomic-numbers '[...]' --E0s '{...}'

  # Quick end-to-end check in your env before submitting:
  python preprocess_parallel.py --selftest
"""

import os

# ---------------------------------------------------------------------------
# MUST run before importing torch / mace: keep per-process virtual memory small
# so the worker fork() does not exceed the kernel commit limit, and keep BLAS /
# torch single-threaded so 32+ workers don't oversubscribe the cores.
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import ast
import glob
import json
import logging
import multiprocessing as mp
import time

import numpy as np
import h5py

# MACE leaf functions -- reused verbatim so the output matches stock preprocessing.
from mace import tools
from mace.data import KeySpecification, update_keyspec_from_kwargs
from mace.data.utils import load_from_xyz, save_configurations_as_HDF5
from mace.tools.scripts_utils import get_atomic_energies
from mace.cli.preprocess_data import pool_compute_stats

# Set by main() before the pools are forked; workers read them as inherited globals
#
# REPO CHANGE: that inheritance only happens under the *fork* start method. It is
# the Linux default up to Python 3.13, but 3.14 switched the default to
# forkserver, under which workers re-import this module and see _KEYSPEC = None
# -- every task then dies with
#   AttributeError: 'NoneType' object has no attribute 'info_keys'
# Pinning the context here keeps the as-run behaviour (fork) on every
# interpreter, instead of depending on a default that has since changed.
_MP = mp.get_context("fork")
_KEYSPEC: KeySpecification = None
_HEAD_NAME = "Default"
_CONFIG_TYPE_WEIGHTS = {"Default": 1.0}


# ===========================================================================
# Phase 1: split an extxyz into N shards by frame (one fast serial text pass)
# ===========================================================================
def split_xyz_roundrobin(input_path: str, shard_paths: list) -> int:
    """Stream `input_path` and round-robin each frame's raw text into shard files.

    Round-robin (frame i -> shard i % N) balances shard sizes without a counting
    pass and without parsing. Returns the total number of frames seen. The frame
    layout is the standard extxyz one used everywhere in this project:
        <n_atoms>\\n <comment>\\n <n_atoms atom lines>   (no blank separators)
    """
    n = len(shard_paths)
    handles = [open(p, "w") for p in shard_paths]
    count = 0
    try:
        with open(input_path, "r") as f:
            while True:
                line1 = f.readline()
                if not line1:
                    break  # EOF
                n_atoms = int(line1.strip())
                comment = f.readline()
                atom_lines = [f.readline() for _ in range(n_atoms)]
                handles[count % n].write(line1 + comment + "".join(atom_lines))
                count += 1
    finally:
        for h in handles:
            h.close()
    return count


# ===========================================================================
# Phase 2: parse one shard with MACE and write its HDF5 (runs in a worker)
# ===========================================================================
def _parse_and_write(task):
    """Worker: shard xyz -> one HDF5 file, using MACE's exact parsing + writer."""
    shard_path, out_path, drop_last, keep_shard = task
    # Silence the per-shard ASE energy/forces/stress-key warnings (we'd get N copies).
    logging.getLogger().setLevel(logging.ERROR)

    # load_from_xyz returns (atomic_energies_dict, configurations). We ignore the
    # dict -- E0s come from the command line and are handled once in main().
    _, configs = load_from_xyz(
        file_path=shard_path,
        key_specification=_KEYSPEC,
        head_name=_HEAD_NAME,
        config_type_weights=_CONFIG_TYPE_WEIGHTS,
        extract_atomic_energies=False,
    )
    with h5py.File(out_path, "w") as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(configs, 0, f)  # 2nd arg unused by MACE

    n = len(configs)
    if not keep_shard:
        os.remove(shard_path)  # targeted file removal of a temp shard we created
    return out_path, n


# ===========================================================================
# Drive one set (train / valid / test) end to end
# ===========================================================================
def process_set(name, dir_name, file_prefix, input_xyz, out_root, shard_dir,
                n_shards, read_procs, keep_shards):
    """Split `input_xyz`, parse shards in parallel, write <out_root>/<dir_name>/."""
    out_dir = os.path.join(out_root, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    shard_paths = [os.path.join(shard_dir, f"{file_prefix}_shard_{i}.xyz")
                   for i in range(n_shards)]

    t0 = time.time()
    count = split_xyz_roundrobin(input_xyz, shard_paths)
    if count == 0:
        for p in shard_paths:
            if os.path.exists(p):
                os.remove(p)
        raise SystemExit(f"[{name}] no frames found in {input_xyz}")
    print(f"[{name}] split {count:,} frames into {n_shards} shards "
          f"in {time.time()-t0:.1f}s", flush=True)

    # Drop empty shards (e.g. when n_shards > n_frames) so we never feed an empty
    # file to load_from_xyz. Match stock's per-set drop_last parity hack exactly.
    nonempty = [p for p in shard_paths if os.path.getsize(p) > 0]
    for p in shard_paths:
        if p not in nonempty and os.path.exists(p):
            os.remove(p)
    drop_last = bool(count % 2 == 1)

    tasks = [(sp, os.path.join(out_dir, f"{file_prefix}_{i}.h5"), drop_last, keep_shards)
             for i, sp in enumerate(nonempty)]

    t0 = time.time()
    with _MP.Pool(processes=min(read_procs, len(tasks))) as pool:
        results = pool.map(_parse_and_write, tasks)
    written = sum(n for _, n in results)
    print(f"[{name}] wrote {written:,} configs to {out_dir}/ "
          f"({len(results)} HDF5 files) in {time.time()-t0:.1f}s", flush=True)

    if written != count:
        print(f"[{name}] WARNING: parsed {written:,} configs but split {count:,} "
              f"frames -- some frames may have failed to parse.", flush=True)
    return count


# ===========================================================================
# Statistics over the produced train HDF5 dir (reuses MACE's pool_compute_stats)
# ===========================================================================
def compute_and_write_statistics(out_root, z_table, r_max, atomic_energies_dict,
                                 batch_size, stats_procs):
    train_dir = os.path.join(out_root, "train")
    atomic_energies = np.array([atomic_energies_dict[z] for z in z_table.zs])

    # Keep torch single-threaded inside each stats worker (fork inherits this).
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    print(f"[stats] computing over {train_dir}/ with {stats_procs} workers ...",
          flush=True)
    t0 = time.time()
    avg_num_neighbors, mean, std = pool_compute_stats(
        [train_dir, z_table, r_max, atomic_energies, batch_size, stats_procs]
    )
    print(f"[stats] avg_num_neighbors={avg_num_neighbors}, mean={mean}, std={std} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # EXACT stock schema (mace/cli/preprocess_data.py): all values stringified the
    # same way so mace_run_train --statistics_file parses it identically.
    statistics = {
        "atomic_energies": str(atomic_energies_dict),
        "avg_num_neighbors": avg_num_neighbors,
        "mean": mean,
        "std": std,
        "atomic_numbers": str([int(z) for z in z_table.zs]),
        "r_max": r_max,
    }
    stats_path = os.path.join(out_root, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(statistics, f)
    print(f"[stats] wrote {stats_path}", flush=True)


# ===========================================================================
# Config / CLI handling
# ===========================================================================
def _normalize_prefix(p):
    # Stock concatenates h5_prefix + "train/...", so 'processed_data/' is a dir
    # prefix. We treat it as a directory root and join paths.
    return p if p else "."


def load_config(path):
    import yaml  # only needed when --config is used
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML (same keys as your MACE preprocess.yaml). "
                                     "CLI flags override yaml values.")
    # Data
    ap.add_argument("--train-file")
    ap.add_argument("--valid-file")
    ap.add_argument("--test-file")
    ap.add_argument("--h5-prefix", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--r-max", type=float, default=None)
    ap.add_argument("--energy-key", default=None)
    ap.add_argument("--forces-key", default=None)
    ap.add_argument("--stress-key", default=None)
    ap.add_argument("--atomic-numbers", default=None,
                    help="String list, e.g. '[1,2,8,...]' (same as MACE).")
    ap.add_argument("--E0s", default=None, help="String dict or 'average' (same as MACE).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train-shards", type=int, default=None,
                    help="Number of train HDF5 files / parallel parse tasks (default 64).")
    ap.add_argument("--valid-shards", type=int, default=None, help="default 16")
    ap.add_argument("--test-shards", type=int, default=None, help="default 16")
    ap.add_argument("--read-procs", type=int, default=None,
                    help="Concurrent parse workers (default 32). Fork-safe with CUDA off.")
    ap.add_argument("--stats-procs", type=int, default=None,
                    help="Concurrent statistics workers (default 32).")
    ap.add_argument("--shard-dir", default=None,
                    help="Where temp xyz shards go (default <h5_prefix>_shards). "
                         "Put it on fast scratch; needs ~one extra copy of each set.")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Do not delete temp xyz shards after writing HDF5.")
    ap.add_argument("--no-stats", action="store_true",
                    help="Skip statistics.json (e.g. valid/test-only runs).")
    ap.add_argument("--selftest", action="store_true",
                    help="Run a tiny synthetic end-to-end check and exit.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}

    def pick(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        if yaml_key in cfg and cfg[yaml_key] is not None:
            return cfg[yaml_key]
        return default

    # num_process from the yaml seeds both pool sizes unless overridden.
    yaml_nproc = cfg.get("num_process")

    args.train_file = pick(args.train_file, "train_file", None)
    args.valid_file = pick(args.valid_file, "valid_file", None)
    args.test_file = pick(args.test_file, "test_file", None)
    args.h5_prefix = _normalize_prefix(pick(args.h5_prefix, "h5_prefix", "processed_data/"))
    args.work_dir = pick(args.work_dir, "work_dir", ".")
    args.r_max = float(pick(args.r_max, "r_max", 6.0))
    args.energy_key = pick(args.energy_key, "energy_key", "energy")
    args.forces_key = pick(args.forces_key, "forces_key", "forces")
    args.stress_key = pick(args.stress_key, "stress_key", "stress")
    args.atomic_numbers = pick(args.atomic_numbers, "atomic_numbers", None)
    args.E0s = pick(args.E0s, "E0s", None)
    args.batch_size = int(pick(args.batch_size, "batch_size", 32))
    args.seed = int(pick(args.seed, "seed", 1))
    args.train_shards = int(pick(args.train_shards, "train_shards", 64))
    args.valid_shards = int(pick(args.valid_shards, "valid_shards", 16))
    args.test_shards = int(pick(args.test_shards, "test_shards", 16))
    args.read_procs = int(pick(args.read_procs, "read_procs", yaml_nproc or 32))
    args.stats_procs = int(pick(args.stats_procs, "stats_procs", yaml_nproc or 32))
    args.shard_dir = pick(args.shard_dir, "shard_dir",
                          os.path.join(args.h5_prefix.rstrip("/") + "_shards"))
    if cfg.get("compute_statistics") is False:
        args.no_stats = True
    return args


def build_keyspec(args):
    # Reproduce stock main() EXACTLY: it does
    # i.e. it feeds the WHOLE args namespace (every *_key with its parser default)
    # into the keyspec. load_from_xyz then indexes info_keys["head"], ["virials"],
    # etc., so all of them must be present -- not just energy/forces/stress.
    # We pull the defaults straight from MACE's own preprocess parser (version-proof)
    # and override only the three keys the user sets.
    parser = tools.build_preprocess_arg_parser()
    keydict = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    keydict["energy_key"] = args.energy_key
    keydict["forces_key"] = args.forces_key
    keydict["stress_key"] = args.stress_key
    ks = KeySpecification()
    update_keyspec_from_kwargs(ks, keydict)
    return ks


# ===========================================================================
# Self-test: tiny synthetic dataset, full pipeline, assert it round-trips
# ===========================================================================
def run_selftest():
    import tempfile
    from ase import Atoms
    from ase.io import write as ase_write

    tmp = tempfile.mkdtemp(prefix="pp_selftest_")
    print(f"[selftest] workdir {tmp}")
    train_xyz = os.path.join(tmp, "train.xyz")

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(7):  # odd count -> exercises drop_last parity
        n = 4
        at = Atoms("H2O2", positions=rng.random((n, 3)) * 3,
                   cell=np.eye(3) * 6, pbc=True)
        at.info["energy"] = float(rng.random())
        at.info["stress"] = rng.random(6)  # ASE extxyz writer expects ndarray (.shape)
        at.arrays["forces"] = rng.random((n, 3))
        frames.append(at)
    ase_write(train_xyz, frames, format="extxyz")

    class A:
        pass
    a = A()
    a.energy_key, a.forces_key, a.stress_key = "energy", "forces", "stress"

    global _KEYSPEC
    _KEYSPEC = build_keyspec(a)
    out_root = os.path.join(tmp, "processed_data")
    shard_dir = os.path.join(tmp, "_shards")

    count = process_set("train", "train", "train", train_xyz, out_root, shard_dir,
                        n_shards=3, read_procs=2, keep_shards=False)
    assert count == 7, count

    z_table = tools.get_atomic_number_table_from_zs([1, 8])
    aedict = get_atomic_energies("{1: -1.0, 8: -2.0}", None, z_table)
    compute_and_write_statistics(out_root, z_table, 5.0, aedict,
                                 batch_size=4, stats_procs=2)

    files = sorted(glob.glob(os.path.join(out_root, "train", "*.h5")))
    total = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            total += len(f["config_batch_0"].keys())
    assert total == 7, total
    with open(os.path.join(out_root, "statistics.json")) as f:
        stats = json.load(f)
    assert set(stats) == {"atomic_energies", "avg_num_neighbors", "mean",
                          "std", "atomic_numbers", "r_max"}, stats
    print(f"[selftest] OK: {len(files)} HDF5 files, {total} configs, "
          f"statistics.json keys verified.")
    print(f"[selftest] statistics.json = {stats}")


# ===========================================================================
def main():
    args = build_args()
    if args.selftest:
        run_selftest()
        return

    if not any([args.train_file, args.valid_file, args.test_file]):
        raise SystemExit("Nothing to do: give at least one of "
                         "--train-file/--valid-file/--test-file.")
    if args.atomic_numbers is None:
        raise SystemExit("--atomic-numbers is required (string list, as in MACE).")

    global _KEYSPEC
    _KEYSPEC = build_keyspec(args)

    zs_list = ast.literal_eval(args.atomic_numbers)
    z_table = tools.get_atomic_number_table_from_zs(zs_list)
    out_root = args.h5_prefix

    print(f"Output root : {out_root}")
    print(f"Shard dir   : {args.shard_dir}")
    print(f"read_procs={args.read_procs}  stats_procs={args.stats_procs}  "
          f"r_max={args.r_max}  CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}'")

    t_all = time.time()
    train_processed = False
    if args.train_file:
        process_set("train", "train", "train", args.train_file, out_root,
                    args.shard_dir, args.train_shards, args.read_procs, args.keep_shards)
        train_processed = True
    if args.valid_file:
        process_set("valid", "val", "val", args.valid_file, out_root,
                    args.shard_dir, args.valid_shards, args.read_procs, args.keep_shards)
    if args.test_file:
        process_set("test", "test", "test", args.test_file, out_root,
                    args.shard_dir, args.test_shards, args.read_procs, args.keep_shards)

    if train_processed and not args.no_stats:
        if args.E0s is None:
            raise SystemExit("--E0s is required to compute statistics (or pass --no-stats).")
        aedict = get_atomic_energies(args.E0s, None, z_table)
        compute_and_write_statistics(out_root, z_table, args.r_max, aedict,
                                     args.batch_size, args.stats_procs)
    elif not train_processed:
        print("No train set -> skipping statistics.json (valid/test-only run). "
              "Reuse a statistics.json produced from your train set.")

    print(f"\nAll done in {(time.time()-t_all)/60:.2f} min -> {out_root}")


if __name__ == "__main__":
    main()
