"""DIRECT step 3a: stack shard_<i>.npy (i = 0..NSHARDS-1) into one feature matrix.

Writes feats.npy and feats.ids.json into the current directory. Rows follow the
shard order, so row r of feats.npy is id r of feats.ids.json. Shards that do
not exist (idle file-mode shards) are skipped with a warning.

    python stack_shards.py <feat_dir> <nshards>
"""

import json
import os
import sys

import numpy as np

outdir, nshards = sys.argv[1], int(sys.argv[2])
npy_paths, ids = [], []
missing = []
for i in range(nshards):
    p = os.path.join(outdir, f"shard_{i}.npy")
    if os.path.exists(p):
        npy_paths.append(p)
    else:
        missing.append(i)
if missing:
    print(f"[stack] WARNING: {len(missing)} shard(s) missing: {missing[:20]}"
          f"{' ...' if len(missing) > 20 else ''}", flush=True)

arrs = [np.load(p, mmap_mode="r") for p in npy_paths]
total = sum(a.shape[0] for a in arrs)
dim = arrs[0].shape[1]
print(f"[stack] {len(arrs)} shards, {total} rows x {dim} dims", flush=True)

feats = np.empty((total, dim), dtype=arrs[0].dtype)
row = 0
for p, a in zip(npy_paths, arrs):
    feats[row:row + a.shape[0]] = a[:]
    row += a.shape[0]
    with open(p.replace(".npy", ".ids.json")) as fh:
        ids.extend(json.load(fh))
np.save("feats.npy", feats)
with open("feats.ids.json", "w") as fh:
    json.dump(ids, fh)
assert len(ids) == total, f"ids ({len(ids)}) != rows ({total})"
print(f"[stack] wrote feats.npy {feats.shape} and feats.ids.json ({len(ids)} ids)", flush=True)
