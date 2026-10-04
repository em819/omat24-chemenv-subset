#!/usr/bin/env python3
"""Shared helpers for the stage-10.5 descriptor analyses (SI S5.2 and S5.3).

Imported by split_descriptors.py, diffuse_descriptors.py and
diffuse_xc_secondshell.py. Needs only numpy, scipy, pandas and pyarrow, so it
runs in env A and env B alike.

What lives here:
  * the analysis-side definitions: a descriptor is analysed if it has >= 30
    atoms (k of the purity kNN), "split" means >= 2 connected components of
    the k=15 kNN graph, "diffuse" means compactness_z_element < 0;
  * EnvTok token parsing (the trailing fields are read from the END, because
    11 polyhedron symbols contain an underscore, e.g. 'BO_1:8');
  * readers for stage 10.3 (meta_shard*.parquet / features_shard*.npy) and
    stage 10.4 (report-card parquet);
  * the stage-10.4 latent space (per-feature z-score + PCA to 50 dims, INVALID
    atoms excluded) and the union kNN-graph components;
  * small, dependency-free statistics: rank AUC, Cramer's V, Benjamini-Hochberg,
    Wilson intervals, a Newton-Raphson logistic regression and eta^2.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MIN_ATOMS = 30          # analysis floor of the manuscript (= k of the purity kNN)
K_GRAPH = 15            # kNN graph of stage 10.4 (--k-graph)
N_PCA = 50              # stage 10.4: --var-keep 0.95 --max-dims 50 -> 50 dims

# <element>_<spin>_<coord_elements>_<symbol>:<cn>_<perturb>_<dmin>_<A>
TOKEN_RE = (r"^(?P<element>[^_]+)_(?P<spin>\d+)_(?P<coord_elements>[^_]+)_"
            r"(?P<ce>.+):(?P<cn>\d+)_(?P<perturb>\d+)_(?P<dmin>\d+)_(?P<A>\d+)$")

_T0 = time.time()


def log(*a):
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


def hdr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


# --------------------------------------------------------------------------- #
# stage 10.4 report card -> one row per analysed descriptor
# --------------------------------------------------------------------------- #
def parse_tokens(ids):
    """Decompose token strings into their seven fields (+ ligand-shell helpers)."""
    s = pd.Series(np.asarray(ids, dtype=object), dtype=object)
    f = s.str.extract(TOKEN_RE)
    bad = f["element"].isna()
    if bad.any():
        raise ValueError(f"unparseable tokens, e.g. {s[bad].head(3).tolist()}")
    for c in ("spin", "cn", "perturb", "dmin", "A"):
        f[c] = f[c].astype(int)
    ligs = f["coord_elements"].str.split("-")
    f["ligands"] = ligs                                        # list, with multiplicity
    f["nb_species"] = ligs.map(lambda L: "-".join(sorted(set(L))))
    f["n_nb_species"] = ligs.map(lambda L: len(set(L)))
    f["has_OF"] = ligs.map(lambda L: ("O" in L) or ("F" in L))
    f.insert(0, "chemenv_id", s.to_numpy())
    return f


def load_report_card(path, min_atoms=MIN_ATOMS):
    """Stage 10.4 report card restricted to the analysed descriptors.

    Only tokens with >= --min-size (20) atoms received the kNN-graph and
    compactness metrics in stage 10.4; of those, the manuscript analyses the
    ones with >= min_atoms atoms. Adds the parsed token fields and the two
    flags used throughout:
      split   : n_components >= 2   (the manuscript's "multiple disjoint regions")
      diffuse : compactness_z_element < 0
    Row order is the parquet's (stage 10.4 sorts by n_atoms, descending); the
    size matching in split_descriptors.py breaks ties in this order.
    """
    rc = pd.read_parquet(path)
    rc = rc[rc["n_components"].notna() & (rc["n_atoms"] >= min_atoms)]
    rc = rc.reset_index(drop=True)
    f = parse_tokens(rc["chemenv_id"])
    if not (f["element"].to_numpy() == rc["element"].to_numpy()).all():
        raise ValueError("report-card 'element' disagrees with the token's first field")
    rc = pd.concat([rc.drop(columns=["element"]), f.drop(columns=["chemenv_id"])], axis=1)
    rc["split"] = rc["n_components"] >= 2
    rc["diffuse"] = rc["compactness_z_element"] < 0
    return rc


# --------------------------------------------------------------------------- #
# stage 10.3 shards
# --------------------------------------------------------------------------- #
def shard_paths(feat_dir):
    fp = sorted(glob.glob(os.path.join(feat_dir, "features_shard*.npy")))
    mp = sorted(glob.glob(os.path.join(feat_dir, "meta_shard*.parquet")))
    if not mp or (fp and len(fp) != len(mp)):
        raise SystemExit(f"no or mismatched stage-10.3 shards under {feat_dir}")
    return fp, mp


def read_meta(feat_dir, columns=("row", "chemenv_id", "element", "sid")):
    """All per-atom metadata rows, with the shard index they came from."""
    _, mp = shard_paths(feat_dir)
    parts = []
    for i, p in enumerate(mp):
        m = pd.read_parquet(p, columns=list(columns))
        m["shard"] = i
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def cell_elements(meta):
    """sid -> '-'-joined sorted element set of the whole cell.

    Built from the per-atom 'element' column, which covers every atom of every
    structure (INVALID-token atoms included), so it is the cell's composition
    without reading any extxyz.
    """
    u = meta[["sid", "element"]].drop_duplicates().sort_values(["sid", "element"])
    return u.groupby("sid", sort=False)["element"].agg("-".join)


def material_of(sid):
    """OMat24 material id = the leading 'agm<digits>' of the structure id."""
    return pd.Series(sid).astype(str).str.extract(r"^(agm\d+)", expand=False)


# --------------------------------------------------------------------------- #
# the stage-10.4 latent space
# --------------------------------------------------------------------------- #
def pca_basis(feat_dir, cache=None, chunk=200_000):
    """Per-feature z-score + PCA, following chemenv_feature_metrics.standardize_and_pca.

    Same pool (all atoms except INVALID tokens) and same standardisation, but
    accumulated in float64 over the memory-mapped shards (stage 10.4 holds the
    data in float32), so the basis agrees with stage 10.4's to floating-point
    precision only; a few kNN graphs with near-tied neighbours can therefore
    differ. Returns (mean, std, V).
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        log(f"PCA basis from cache {cache}: {float(z['var_pct']):.1f}% variance")
        return z["mean"], z["std"], z["V"]
    fp, mp = shard_paths(feat_dir)
    n, sx, sxx = 0, None, None
    for f, m in zip(fp, mp):
        valid = ~pd.read_parquet(m, columns=["chemenv_id"])["chemenv_id"].str.contains(
            "INVALID", na=False).to_numpy()
        mm = np.load(f, mmap_mode="r")
        if sx is None:
            sx, sxx = np.zeros(mm.shape[1]), np.zeros((mm.shape[1],) * 2)
        for s in range(0, mm.shape[0], chunk):
            Z = np.asarray(mm[s:s + chunk], dtype=np.float64)[valid[s:s + chunk]]
            sx += Z.sum(0)
            sxx += Z.T @ Z
            n += len(Z)
    mean = sx / n
    cov = sxx / n - np.outer(mean, mean)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    std[std < 1e-12] = 1.0
    evals, evecs = np.linalg.eigh(cov / np.outer(std, std))
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None)
    V = evecs[:, order[:N_PCA]].astype(np.float32)
    var_pct = 100 * evals[:N_PCA].sum() / evals.sum()
    log(f"PCA {len(mean)} -> {N_PCA} dims over {n:,} atoms: {var_pct:.1f}% variance")
    if cache:
        np.savez(cache, mean=mean, std=std, V=V, var_pct=var_pct)
    return mean, std, V


def gather_latent(feat_dir, meta, ids, basis):
    """PCA-50 coordinates of every atom whose token is in `ids`.

    Returns (sub-meta with a fresh index, float32 array aligned to it).
    """
    mean, std, V = basis
    fp, _ = shard_paths(feat_dir)
    sub = meta[meta["chemenv_id"].isin(set(ids))]
    parts, frames = [], []
    for i, f in enumerate(fp):
        s = sub[sub["shard"] == i]
        if len(s):
            raw = np.asarray(np.load(f, mmap_mode="r")[s["row"].to_numpy()], np.float32)
            parts.append((((raw - mean) / std) @ V).astype(np.float32))
            frames.append(s)
    return pd.concat(frames, ignore_index=True), np.vstack(parts)


def knn_components(X, k=K_GRAPH):
    """Connected components of the symmetric (union) kNN graph, as stage 10.4."""
    n = len(X)
    kk = min(k, n - 1)
    _, idx = cKDTree(X).query(X, k=kk + 1)
    rows = np.repeat(np.arange(n), kk)
    A = csr_matrix((np.ones(rows.size, bool), (rows, idx[:, 1:].ravel())), shape=(n, n))
    return connected_components(A.maximum(A.T), directed=False)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def mwu_auc(a, b):
    """Two-sided Mann-Whitney U test; AUC = P(a > b) + P(a = b)/2."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    return r.statistic / (len(a) * len(b)), r.pvalue


def cramers_v(x, y):
    """chi^2 test of a contingency table, Cramer's V and its bias-corrected form.

    Returns (chi2, dof, p, V, V_corrected); V_corrected is Bergsma's (2013)
    small-sample correction, which removes the upward bias many-level
    variables get by chance.
    """
    t = pd.crosstab(pd.Series(np.asarray(x)), pd.Series(np.asarray(y))).to_numpy(float)
    chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
    n = t.sum()
    r, c = t.shape
    v = np.sqrt(chi2 / (n * (min(r, c) - 1)))
    phi2c = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    rc_, cc_ = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    vc = np.sqrt(phi2c / min(rc_ - 1, cc_ - 1))
    return chi2, dof, p, v, vc


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval of a binomial proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def logit(y, X, names, maxiter=100, tol=1e-10):
    """Maximum-likelihood logistic regression (Newton-Raphson, intercept added).

    Returns a dict with coefficient table (coef, se, z, p, odds ratio), the
    log-likelihood, the null log-likelihood and McFadden's pseudo-R^2.
    Raises if the fit does not converge (e.g. complete separation).
    """
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    beta = np.zeros(X.shape[1])

    def ll(b):
        eta = X @ b
        return float(np.sum(y * eta - np.logaddexp(0, eta)))

    cur = ll(beta)
    for _ in range(maxiter):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        t = 1.0
        while True:                                    # step halving keeps ll monotone
            new = ll(beta + t * step)
            if new >= cur - 1e-12 or t < 1e-8:
                break
            t /= 2
        beta = beta + t * step
        done = abs(new - cur) < tol
        cur = new
        if done:
            break
    else:
        raise RuntimeError("logistic regression did not converge")
    mu = 1 / (1 + np.exp(-(X @ beta)))
    cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    se = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(se)) or se.max() > 1e3:
        raise RuntimeError("logistic regression is (quasi-)separated")
    ybar = y.mean()
    llnull = len(y) * (ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))
    tab = pd.DataFrame({"coef": beta, "se": se}, index=["const"] + list(names))
    tab["z"] = tab.coef / tab.se
    tab["p"] = 2 * stats.norm.sf(np.abs(tab.z))
    tab["odds_ratio"] = np.exp(tab.coef)
    return dict(table=tab, llf=cur, llnull=llnull, prsquared=1 - cur / llnull,
                df_model=X.shape[1] - 1)


def lr_test(full, reduced):
    """Likelihood-ratio test of two nested logit() fits: (chi2, df, p)."""
    lr = 2 * (full["llf"] - reduced["llf"])
    df = full["df_model"] - reduced["df_model"]
    return lr, df, stats.chi2.sf(max(lr, 0.0), df)


def _eta2(Xc, tot, codes):
    """eta^2 for integer group codes, given centred X and its total sum of squares."""
    G = csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
    S = np.asarray(G @ Xc)                       # per-group sums of the centred X
    return float(((S ** 2).sum(1) / np.bincount(codes)).sum()) / tot


def eta2(X, groups):
    """Between-group share of the total sum of squared distances to the centroid."""
    Xc = np.asarray(X, np.float64) - np.asarray(X, np.float64).mean(0)
    return _eta2(Xc, float((Xc ** 2).sum()), pd.factorize(np.asarray(groups))[0])


def eta2_perm(X, groups, rng, nperm=200):
    """eta^2 with a label-permutation null: (observed, null mean, z, one-sided p)."""
    Xc = np.asarray(X, np.float64) - np.asarray(X, np.float64).mean(0)
    tot = float((Xc ** 2).sum())
    codes = pd.factorize(np.asarray(groups))[0]
    obs = _eta2(Xc, tot, codes)
    null = np.array([_eta2(Xc, tot, rng.permutation(codes)) for _ in range(nperm)])
    sd = null.std()
    z = (obs - null.mean()) / sd if sd > 0 else np.nan
    return obs, float(null.mean()), z, (1 + int((null >= obs).sum())) / (nperm + 1)
