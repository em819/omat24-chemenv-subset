#!/usr/bin/env python3
"""Stage 10.5b -- what are the latent-space neighbours that carry a DIFFERENT descriptor?

Manuscript statements (main text, "Mapping the EnvTok descriptor to MLFF latent
space", and SI S5.1):
    "For 40,000 atoms drawn uniformly at random from the atoms of these
     sufficiently frequent EnvTok descriptors, we determined the k=30 nearest
     neighbors among all atoms of the validation set. Of the neighbours carrying
     a different descriptor, 99.9 % share the central element, 69.7 % the same
     set of ligand elements and 56.9 % the same coordination number."

Procedure
  1. Load the stage 10.3 per-atom features (features_shard*.npy) and tokens
     (meta_shard*.parquet). Atoms whose token contains INVALID are dropped, as
     stage 10.4 drops them.
  2. Build the latent space of stage 10.4 (--pca stage10.4, default): per-feature
     standardisation, then projection on the leading principal components
     (--var-keep 0.95, capped at --max-dims 50: 50 dims, 94.7 % of the variance
     on the published data). Same arithmetic and dtypes as stage 10.4's
     standardize_and_pca, so the neighbours are those behind its kNN purity.
  3. Characterised descriptors = tokens with >= --min-atoms (30) atoms. The
     threshold applies only to the QUERIED descriptors (their statistics need
     enough atoms); the neighbour pool is every valid atom, however rare its
     token, exactly as for the kNN purity of stage 10.4.
  4. Draw --n-query atoms uniformly, without replacement, from the atoms of the
     characterised descriptors (numpy default_rng(--seed); the published run used
     seed 3) and find their --k exact Euclidean nearest neighbours in the full
     pool (faiss IndexFlatL2). The query atom itself is removed by identity.
  5. A neighbour is "foreign" if its token differs from the query's. Over all
     foreign (query, neighbour) pairs, pooled, report the share whose token has
     the same central element, the same SET of ligand elements (distinct
     elements of the neighbour list, stoichiometry ignored) and the same
     coordination number. All three fields are parsed from the token strings.

--pca original-float64 reproduces the geometry of the first (out-of-repo)
version of this analysis: a float64 PCA fitted on ALL atoms including the 542
INVALID ones (population covariance), projected in float64. It moves the three
percentages by less than the quoted precision; use it only to reproduce those
exact digits.

Environment: envB-mace (numpy, pandas, pyarrow, faiss). Peak RAM ~9 GB: the features
are streamed from the memmaps; tokens and the 50-d projection stay in RAM. The exact
search of 40,000 queries against 12 M atoms takes
~25 min on 14 CPU cores; seconds with --knn-backend faiss-gpu.
"""

import argparse
import glob
import json
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import faiss

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --feature-glob / --meta-glob  REQUIRED. Stage 10.3's shards, e.g.
#                  '../3_extract_features/mace_features/features_shard*.npy' and
#                  '../3_extract_features/mace_features/meta_shard*.parquet'
#                  (the same globs submit_metrics.sh passes to stage 10.4).
#   --report-card  OPTIONAL. Stage 10.4's report-card parquet; if given, the set
#                  of characterised descriptors is checked against its tokens
#                  with n_atoms >= --min-atoms.
#   --var-keep / --max-dims  must equal stage 10.4's (published: 0.95 / 50).
#   --seed         query draw; the published numbers used 3.
#   --knn-backend  faiss (CPU) or faiss-gpu. Both are exact.
#   --out-json     OPTIONAL. Write all printed numbers to this JSON file.
# ============================================================================

# {central}_{spin}_{neighbour list}_{CE symbol}:{CN}_{csm bin}_{dmin bin}_{A bin}.
# The CE symbol may itself contain '_' ('BO_1:8'), so it is matched greedily
# between the neighbour list and the ':'; no other field contains ':'.
RE_TOKEN = re.compile(r"^(?P<element>[^_]+)_(?P<spin>\d+)_(?P<nb>[^_]+)_"
                      r"(?P<ce>.+):(?P<cn>\d+)_(?P<csm>\d+)_(?P<dmin>\d+)_(?P<A>\d+)$")

_t0 = time.time()


def log(*a):
    print(f"[{time.time() - _t0:7.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------
# Loading. Features are never held in RAM as a whole: every pass streams them
# from the stage 10.3 memmaps (the 12 M x 50 projection itself is ~2.4 GB).
# ----------------------------------------------------------------------------
def load_meta(feature_glob, meta_glob):
    """Shard paths, INVALID mask over all atoms, and for the valid atoms (in shard
    order) integer token codes, the token strings, and the meta element column."""
    fpaths, mpaths = sorted(glob.glob(feature_glob)), sorted(glob.glob(meta_glob))
    if not fpaths or len(fpaths) != len(mpaths):
        raise SystemExit(f"feature/meta shard mismatch: {len(fpaths)} vs {len(mpaths)}")
    tabs = []
    for fp, mp in zip(fpaths, mpaths):
        t = pq.read_table(mp, columns=["row", "chemenv_id", "element"])
        n_rows = np.load(fp, mmap_mode="r").shape[0]
        # stage 10.3 writes row = 0..n-1, aligned with the feature rows
        if t.num_rows != n_rows or not np.array_equal(t["row"].to_numpy(), np.arange(n_rows)):
            raise SystemExit(f"{mp}: rows not aligned with {fp}")
        tabs.append(t.select(["chemenv_id", "element"]))
    T = pa.concat_tables(tabs)
    valid = pc.invert(pc.match_substring(T["chemenv_id"], "INVALID")).to_numpy()
    log(f"{len(valid):,} atoms in {len(fpaths)} shards; {(~valid).sum():,} INVALID")
    T = T.filter(pa.array(valid))
    tok = T["chemenv_id"].combine_chunks().dictionary_encode()
    el = T["element"].combine_chunks().dictionary_encode()
    return (fpaths, valid,
            tok.indices.to_numpy().astype(np.int64),
            np.asarray(tok.dictionary.to_pylist(), dtype=object),
            np.asarray(el.dictionary.to_pylist(), dtype=object)[el.indices.to_numpy()])


def feature_blocks(fpaths, keep, size, dtype=np.float32):
    """Consecutive blocks of exactly `size` kept rows (the last one shorter), in
    shard order and across shard boundaries: block i equals X[i*size:(i+1)*size]
    of the concatenated array of kept rows."""
    buf, nbuf, g = [], 0, 0
    for fp in fpaths:
        mm = np.load(fp, mmap_mode="r")
        for s in range(0, mm.shape[0], size):
            e = min(s + size, mm.shape[0])
            buf.append(np.asarray(mm[s:e], dtype=dtype)[keep[g + s:g + e]])
            nbuf += len(buf[-1])
            if nbuf >= size:
                cat = np.concatenate(buf)
                yield cat[:size]
                buf, nbuf = [cat[size:]], len(cat) - size
        g += mm.shape[0]
    if nbuf:
        yield np.concatenate(buf)


# ----------------------------------------------------------------------------
# Latent space
# ----------------------------------------------------------------------------
def _seq_sum_f32(blocks, f=None):
    """Column sums in float32, accumulated row after row -- the order in which
    numpy reduces axis 0 of a C-contiguous float32 array -- so the result equals
    X.sum(axis=0) of the concatenated blocks bit for bit."""
    acc = None
    for b in blocks:
        if f is not None:
            b = f(b)
        if acc is not None:
            b = np.vstack([acc[None, :], b])
        acc = np.add.reduce(b, axis=0)
    return acc


def pca_stage10_4(fpaths, valid, var_keep, max_dims, chunk=200_000):
    """Stage 10.4's standardize_and_pca (cosine=False) on the valid atoms, line for
    line, with X streamed in the same 200,000-row chunks. X.mean(axis=0) and
    X.std(axis=0) are reproduced bit for bit by sequential float32 sums."""
    blocks = lambda: feature_blocks(fpaths, valid, chunk)
    n, d = int(valid.sum()), np.load(fpaths[0], mmap_mode="r").shape[1]
    mean = np.true_divide(_seq_sum_f32(blocks()), n, dtype=np.float32)
    std = np.sqrt(np.true_divide(_seq_sum_f32(blocks(), f=lambda b: np.square(b - mean)),
                                 n, dtype=np.float32))
    std[std < 1e-12] = 1.0

    C = np.zeros((d, d), dtype=np.float64)
    for X in blocks():
        Z = (X - mean) / std
        C += Z.T @ Z
    C /= max(n - 1, 1)

    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals[evals < 0] = 0.0
    cumvar = np.cumsum(evals) / max(evals.sum(), 1e-12)
    k = int(np.searchsorted(cumvar, var_keep) + 1)
    k = max(1, min(k, max_dims, d))
    V = evecs[:, :k].astype(np.float32)

    Xp = np.empty((n, k), dtype=np.float32)
    s = 0
    for X in blocks():
        Z = ((X - mean) / std).astype(np.float32)
        Xp[s:s + len(X)] = Z @ V
        s += len(X)
    return Xp, k, float(cumvar[k - 1])


def pca_original_float64(fpaths, valid, n_dims, chunk=200_000):
    """The geometry of the first version of this analysis: float64 moments over
    ALL atoms (INVALID included), eigenvectors of the correlation matrix, the
    leading n_dims kept, valid atoms projected in float64."""
    D = np.load(fpaths[0], mmap_mode="r").shape[1]
    n, sx, sxx = 0, np.zeros(D), np.zeros((D, D))
    for fp in fpaths:
        mm = np.load(fp, mmap_mode="r")
        for s in range(0, mm.shape[0], chunk):
            Z = np.asarray(mm[s:s + chunk], dtype=np.float64)
            sx += Z.sum(0)
            sxx += Z.T @ Z
            n += len(Z)
    mean = sx / n
    cov = sxx / n - np.outer(mean, mean)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    std[std < 1e-12] = 1.0
    evals, evecs = np.linalg.eigh(cov / np.outer(std, std))
    o = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[o], 0, None), evecs[:, o]
    V = evecs[:, :n_dims].astype(np.float32)
    Xp = np.empty((int(valid.sum()), n_dims), dtype=np.float32)
    s = 0
    for X in feature_blocks(fpaths, valid, chunk):
        Xp[s:s + len(X)] = ((X - mean) / std) @ V
        s += len(X)
    return Xp, n_dims, float(evals[:n_dims].sum() / evals.sum())


# ----------------------------------------------------------------------------
# kNN (exact)
# ----------------------------------------------------------------------------
def knn_excluding_self(Xp, q, k, backend):
    """k exact nearest neighbours of Xp[q] among ALL rows of Xp, the query atom
    removed by identity. (Removing column 0 instead would be wrong when a
    symmetry-equivalent twin at distance 0 is returned before the atom itself.)"""
    index = faiss.IndexFlatL2(Xp.shape[1])
    if backend == "faiss-gpu":
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(Xp)
    _, I = index.search(np.ascontiguousarray(Xp[q]), k + 1)
    is_self = I == q[:, None]
    absent = ~is_self.any(1)          # > k zero-distance twins pushed self out:
    is_self[absent, -1] = True        # then drop the farthest of the k+1 instead
    log(f"  self first in the list for {100 * is_self[:, 0].mean():.2f} % of queries; "
        f"absent for {int(absent.sum())}")
    return I[~is_self].reshape(len(q), k)


# ----------------------------------------------------------------------------
# Token fields
# ----------------------------------------------------------------------------
def token_fields(tokens):
    """Central element, ligand-element set and CN of each token string."""
    F = pd.Series(tokens, dtype=object).str.extract(RE_TOKEN)
    if F["element"].isna().any():
        bad = list(pd.Series(tokens)[F["element"].isna()][:3])
        raise SystemExit(f"{int(F['element'].isna().sum())} unparseable tokens, e.g. {bad}")
    nb = F["nb"].str.split("-")
    cn = F["cn"].astype(int).to_numpy()
    # the neighbour list has one entry per coordinating atom
    assert (nb.map(len).to_numpy() == cn).all(), "CN != length of the neighbour list"
    return dict(elem=F["element"].to_numpy(object),
                lig=nb.map(lambda L: "-".join(sorted(set(L)))).to_numpy(object),
                cn=cn)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-glob", required=True)
    ap.add_argument("--meta-glob", required=True)
    ap.add_argument("--report-card", default=None)
    ap.add_argument("--min-atoms", type=int, default=30,
                    help="characterised (queried) descriptors: at least this many atoms")
    ap.add_argument("--k", type=int, default=30, help="neighbours per query atom")
    ap.add_argument("--n-query", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--var-keep", type=float, default=0.95)
    ap.add_argument("--max-dims", type=int, default=50)
    ap.add_argument("--pca", choices=["stage10.4", "original-float64"], default="stage10.4")
    ap.add_argument("--knn-backend", choices=["faiss", "faiss-gpu"], default="faiss")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    print(f"settings: {vars(a)}", flush=True)

    fpaths, valid, code, cats, elem_meta = load_meta(a.feature_glob, a.meta_glob)
    if a.pca == "stage10.4":
        Xp, ndim, var = pca_stage10_4(fpaths, valid, a.var_keep, a.max_dims)
    else:
        Xp, ndim, var = pca_original_float64(fpaths, valid, a.max_dims)
    log(f"PCA ({a.pca}): {Xp.shape[0]:,} valid atoms -> {ndim} dims, "
        f"{100 * var:.2f} % variance")

    counts = np.bincount(code, minlength=len(cats))
    char_tok = counts >= a.min_atoms
    char_idx = np.where(char_tok[code])[0]
    log(f"{len(cats):,} distinct tokens; characterised (>= {a.min_atoms} atoms): "
        f"{int(char_tok.sum()):,} tokens, {len(char_idx):,} atoms")
    if a.report_card:
        rc = pd.read_parquet(a.report_card, columns=["chemenv_id", "n_atoms"])
        same = set(rc.loc[rc["n_atoms"] >= a.min_atoms, "chemenv_id"]) == set(cats[char_tok])
        log(f"characterised set == report-card tokens with n_atoms >= {a.min_atoms}: {same}")
        if not same:
            raise SystemExit("token sets differ: features and report card do not match")

    rng = np.random.default_rng(a.seed)
    q = char_idx[rng.choice(len(char_idx), size=min(a.n_query, len(char_idx)),
                            replace=False)]
    log(f"exact kNN: {len(q):,} queries x {len(Xp):,} atoms, k={a.k} ({a.knn_backend}) ...")
    NB = knn_excluding_self(Xp, q, a.k, a.knn_backend)
    log("kNN done")

    # Parse only the tokens that occur among queries and neighbours.
    used = np.unique(np.concatenate([code[q], code[NB].ravel()]))
    fld_used = token_fields(cats[used])
    pos = np.full(len(cats), -1)
    pos[used] = np.arange(len(used))
    atoms_used = np.unique(np.concatenate([q, NB.ravel()]))
    if not (fld_used["elem"][pos[code[atoms_used]]] == elem_meta[atoms_used]).all():
        raise SystemExit("central element in the token != element column of the meta")

    qc, nc = pos[code[q]][:, None], pos[code[NB]]          # rows into fld_used
    same_tok = code[q][:, None] == code[NB]
    foreign = ~same_tok
    nf = int(foreign.sum())
    res = dict(settings=vars(a), pca_dims=ndim, pca_variance=var,
               n_atoms=int(len(Xp)), n_tokens=int(len(cats)),
               n_char_tokens=int(char_tok.sum()), n_char_atoms=int(len(char_idx)),
               n_query=int(len(q)), knn_purity_query_mean=float(same_tok.mean()),
               foreign_pairs=nf,
               foreign_from_noncharacterised_pct=float(
                   100 * (foreign & ~char_tok[code[NB]]).sum() / max(nf, 1)),
               pooled_pct={}, per_query_token_pct={})

    print(f"\n  {len(q):,} query atoms, k={a.k}: kNN purity {same_tok.mean():.4f}; "
          f"foreign neighbour pairs {nf:,} of {same_tok.size:,} "
          f"({100 * nf / same_tok.size:.1f} %), "
          f"{res['foreign_from_noncharacterised_pct']:.1f} % of them from tokens "
          f"with < {a.min_atoms} atoms")
    print("\n  foreign neighbours sharing ...        pooled over pairs   "
          "(mean over query tokens)")
    qtok = np.repeat(code[q], a.k)[foreign.ravel()]
    paper = {"elem": 99.9, "lig": 69.7, "cn": 56.9}
    for key, label in [("elem", "the central element"),
                       ("lig", "the ligand-element set"),
                       ("cn", "the coordination number")]:
        m = (fld_used[key][qc] == fld_used[key][nc]) & foreign
        pooled = 100 * m.sum() / nf
        per_tok = 100 * pd.Series(m[foreign]).groupby(qtok).mean().mean()
        res["pooled_pct"][key], res["per_query_token_pct"][key] = float(pooled), float(per_tok)
        print(f"    {label:30s} {pooled:8.2f} %          ({per_tok:6.2f} %)"
              f"    [paper: {paper[key]} %]")

    if a.out_json:
        with open(a.out_json, "w") as fh:
            json.dump(res, fh, indent=2)
        log(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
