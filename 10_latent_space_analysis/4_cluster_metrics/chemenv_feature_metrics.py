#!/usr/bin/env python3

"""
Stage 2: cheap full-coverage test of whether `chemenv_id` structures the MACE
feature space. Consumes the per-atom features written by extract_mace_features.py
and produces a per-chemenv_id "report card".

This is the go/no-go prototype recommended before investing in HDBSCAN /
persistent-homology tooling. It computes three metrics per chemenv_id:

  (1) kNN label purity  -- for each atom, the fraction of its k nearest
      neighbours (global pool, in reduced feature space) that share its
      chemenv_id; averaged per id. The standard non-parametric representation
      probe (the "kNN probe" used to score DINO/SSL embeddings). Captures
      cohesion AND separation in one number in [0,1]. (FAISS-free: scipy cKDTree.)

  (2) compactness vs a size-matched NULL -- RMS radius of the class about its
      centroid, divided by the global RMS radius (dispersion_ratio), plus a
      z-score against random subsets of the SAME size drawn from the whole
      pool (compactness_z). Positive z => tighter than chance. This is the only
      metric calibrated against "what a class of this size looks like by
      accident", which matters because class sizes vary enormously here.

  (3) kNN-graph connected components -- build the SYMMETRIC (union) k-NN graph
      on the class's own points and count connected components
      (scipy.sparse.csgraph). n_components_massive > 1 (>=2 components each
      holding >=10% of the points) with a small largest_comp_frac => the
      chemenv_id spans multiple DISJOINT regions of feature space (multimodal);
      ~1 big component => one connected region. This is the scalable sibling of
      HDBSCAN / 0-dim persistent homology; use it to FLAG multimodal ids for the
      heavier topological pass later. (Union, not mutual-kNN: mutual-kNN
      over-fragments even single blobs -- see knn_graph_components.)

DEPENDENCIES: numpy, scipy, pandas, pyarrow, joblib -> all in the pymatgen env.
OPTIONAL: faiss (faiss-cpu / faiss-gpu) for a 10-100x faster EXACT global kNN;
falls back to scipy cKDTree automatically if faiss is absent. The per-class
metrics fan out over CPU cores with joblib (--n-jobs); output is bit-identical
to the serial run at any core count (per-class subsampling is seeded by class id).

DATA MODEL: each shard from stage 1 is features_shard{ID}.npy (float32 memmap,
rows = atoms) aligned row-for-row with meta_shard{ID}.parquet (chemenv_id,
f_ae_mace, element, sid, local_idx). We load ALL metadata (small), optionally
take a stratified subsample to --max-atoms, then gather ONLY those feature rows
from the memmaps so peak RAM is bounded by the subsample.

USAGE
  Real data:
    python chemenv_feature_metrics.py \
        --feature-glob 'mace_features/features_shard*.npy' \
        --meta-glob    'mace_features/meta_shard*.parquet' \
        --out report_card.parquet --k 30 --max-atoms 1500000
  Self-test (no real data; fabricates labelled blobs and checks the metrics):
    python chemenv_feature_metrics.py --selftest
"""

import argparse
import glob
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from joblib import Parallel, delayed


def _safe_spearman(x, y):
    """Spearman rho, NaN-robust: needs >=5 finite pairs and non-constant inputs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    xm, ym = x[m], y[m]
    if np.ptp(xm) == 0 or np.ptp(ym) == 0:
        return float("nan")
    r = spearmanr(xm, ym)[0]
    return float(r) if r == r else float("nan")


def coord_number_from_id(cid):
    """Coordination number from a chemenv_id.

    Token layout produced by analyze_*_twocutoffs_dists_and_A.py:
        {central}_{spin}_{coords}_{symbol}_{pert}_{rmin_bin}_{a_bin}
    (the _dists layout it replaces ended in {rmin_bin}_{rmax_bin})
    The pre-_dists token ended at {pert}, so the original code read the geometry
    symbol as the 2nd-to-last '_' field. With the two distance-bin fields appended
    that position now holds {rmin_bin} -- a bare integer with no ':' -- and the
    original code silently returned -1 for EVERY atom.

    Anchor on the ':' instead. The geometry symbol is the only field that can carry
    one ('O:6', 'S:1', 'UNKNOWN:4'): {central} and {coords} are element symbols and
    '-' separators, and {spin}/{pert}/{rmin_bin}/{rmax_bin} are integers. That makes
    this position-independent, so it also reads the pre-_dists 5-field token
    identically, and it stays correct for the 11 chemenv symbols that contain an
    underscore themselves ('BO_1:8', 'TT_2:9', 'BS_1:10', ...), which shift every
    later field by one.

    _and_A: UNAFFECTED, deliberately verified rather than assumed. The last field is
    now {a_bin} instead of {rmax_bin}, and {rmin_bin} now has five states (0..4)
    rather than four, but this function reads neither -- it anchors on the ':' of
    the geometry symbol, and no other field in the token can contain a ':'. Checked
    in test_and_A_downstream.py against real _and_A tokens including
    underscore-bearing symbols, 'UNKNOWN:n' and 'INVALID'.

    Returns the integer n, or -1 if unparseable ('INVALID', bare 'UNKNOWN')."""
    parts = str(cid).split("_")
    for sym in parts:
        if ":" in sym:
            try:
                return int(sym.split(":")[1])
            except (ValueError, IndexError):
                return -1
    return -1


def _pearson(df, xcol, ycol):
    sub = df[[xcol, ycol]].dropna()
    if len(sub) <= 3:
        return float("nan"), len(sub)
    return float(np.corrcoef(sub[xcol], sub[ycol])[0, 1]), len(sub)


def compactness_error_report(big, plot=True, plot_out=None):
    """Between-token corr(compactness_z, force error) under confound filters, + a scatter.

    Filtering on compactness_z restricts the x-range and ATTENUATES r by construction;
    the CN>=3 filter (dropping under-coordinated, under-specified tokens) is the cleaner
    de-confounder because it does not touch the x-axis. Both are reported so the
    range-restriction effect is visible.
    """
    big = big.copy()
    big["coordination"] = [coord_number_from_id(c) for c in big["chemenv_id"]]
    cn = big["coordination"].to_numpy()
    z = big["compactness_z"].to_numpy()
    ytargets = [("mean_force_error", "abs")]
    if "mean_rel_force_error" in big.columns:
        ytargets.append(("mean_rel_force_error", "rel"))

    print("\n=== compactness_z vs force error (confound-filtered) ===")
    print("  NOTE: filtering on compactness_z truncates the x-range -> attenuates r by "
          "construction.\n        The CN>=3 filter is the cleaner de-confounder "
          "(removes under-coordinated, under-specified tokens without touching x).")
    filters = [
        ("all tokens", np.ones(len(big), bool)),
        ("CN>=3 (drop undercoordinated)", cn >= 3),
        ("CN>=4", cn >= 4),
        ("compactness_z>=0", z >= 0),
        ("CN>=3 & z>=0", (cn >= 3) & (z >= 0)),
    ]
    for ycol, lab in ytargets:
        print(f"  vs {ycol}:")
        for fname, mask in filters:
            r, nn = _pearson(big[mask], "compactness_z", ycol)
            print(f"     {fname:32s} r={r:+.3f}  (n={nn})")

    if plot:
        _plot_compactness_error(big, ytargets, plot_out)


def _plot_compactness_error(big, ytargets, plot_out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (plot skipped: matplotlib unavailable: {e})")
        return
    ycol, lab = ytargets[-1]                    # prefer relative error if present
    sub = big[["compactness_z", ycol, "coordination", "n_atoms"]].dropna()
    sub = sub[sub[ycol] > 0]
    if len(sub) < 5:
        print("  (plot skipped: too few finite points)")
        return
    x = sub["compactness_z"].to_numpy()
    y = sub[ycol].to_numpy()
    c = sub["coordination"].to_numpy().clip(0, 12)
    s = 3 + 5 * np.log10(np.clip(sub["n_atoms"].to_numpy(), 1, None))
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(np.clip(x, -120, 60), y, c=c, s=s, alpha=0.35,
                    cmap="viridis", linewidths=0)
    ax.set_xlim(-122, 62)
    ax.set_yscale("log")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("compactness_z  (x clipped to [-120,60]; negative = diffuse)")
    ax.set_ylabel(f"{ycol}  (log scale)")
    cb = fig.colorbar(sc); cb.set_label("coordination number")
    rr, nn = _pearson(sub, "compactness_z", ycol)
    ax.set_title(f"token compactness vs {lab} force error   (all tokens r={rr:+.3f}, n={nn})")
    fig.tight_layout()
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"  wrote plot -> {plot_out}")

def frequency_error_report(big, plot=True, plot_out=None):
    """Learning-curve test: does token frequency (n_atoms, a training-coverage proxy)
    predict force error across the analyzable head?

    The head is past the knee of the error~A*n^-alpha + eps_floor learning curve, so the
    expectation is a ~FLAT relationship: error there is set by the irreducible floor
    (label noise + target complexity + force magnitude), not by sample count. Any genuine
    data-starvation signal lives in the rare tail, which --min-size excludes. We report the
    Spearman of log10(n_atoms) vs error (the quantitative 'frequency<->error correlation'),
    incl. a CN>=3 variant that drops under-coordinated tokens, plus a binned curve.
    """
    big = big.copy()
    if "coordination" not in big.columns:
        big["coordination"] = [coord_number_from_id(c) for c in big["chemenv_id"]]
    logn = np.log10(big["n_atoms"].to_numpy().astype(float))
    cn = big["coordination"].to_numpy()
    ytargets = [("mean_force_error", "abs")]
    if "mean_rel_force_error" in big.columns:
        ytargets.append(("mean_rel_force_error", "rel"))

    print("\n=== frequency (n_atoms, training-coverage proxy) vs force error ===")
    print("  NOTE: the analyzable head is past the learning-curve knee, so a FLAT relation\n"
          "        is expected -- data-starvation lives in the rare tail (excluded by --min-size).")
    print("  Spearman(log10 n_atoms, error) across tokens:")
    for ycol, _ in ytargets:
        rho_all = _safe_spearman(logn, big[ycol].to_numpy())
        m3 = cn >= 3
        rho_cn = _safe_spearman(logn[m3], big[ycol].to_numpy()[m3])
        print(f"     vs {ycol:24s} rho={rho_all:+.3f} (n={np.isfinite(big[ycol]).sum()})"
              f"   [CN>=3] rho={rho_cn:+.3f} (n={int((m3 & np.isfinite(big[ycol])).sum())})")

    # Binned learning curve: log-spaced frequency bins, per-token means averaged within bin.
    nmin, nmax = big["n_atoms"].min(), big["n_atoms"].max()
    edges = np.unique(np.round(np.geomspace(nmin, nmax + 1, 9)).astype(int))
    if len(edges) >= 3:
        binid = np.digitize(big["n_atoms"].to_numpy(), edges[1:-1])
        print("  Binned (per-token means averaged within each frequency bin):")
        hdr = f"     {'n_atoms range':>18s}  {'n_tokens':>8s}  {'atoms':>10s}  {'mean_abs_err':>12s}"
        if "mean_rel_force_error" in big.columns:
            hdr += f"  {'mean_rel_err':>12s}"
        print(hdr)
        for b in range(len(edges) - 1):
            sel = big[binid == b]
            if len(sel) == 0:
                continue
            lo, hi = edges[b], edges[b + 1]
            line = (f"     {f'[{lo}, {hi})':>18s}  {len(sel):>8d}  "
                    f"{int(sel['n_atoms'].sum()):>10d}  {sel['mean_force_error'].mean():>12.2f}")
            if "mean_rel_force_error" in big.columns:
                line += f"  {sel['mean_rel_force_error'].mean():>12.4f}"
            print(line)

    if plot:
        _plot_frequency_error(big, ytargets, plot_out)


def _plot_frequency_error(big, ytargets, plot_out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (frequency plot skipped: matplotlib unavailable: {e})")
        return
    fig, axes = plt.subplots(1, len(ytargets), figsize=(6 * len(ytargets), 5), squeeze=False)
    for ax, (ycol, lab) in zip(axes[0], ytargets):
        sub = big[["n_atoms", ycol]].dropna()
        sub = sub[sub[ycol] > 0]
        if len(sub) < 5:
            ax.set_title(f"{lab}: too few points"); continue
        x = sub["n_atoms"].to_numpy().astype(float)
        y = sub[ycol].to_numpy()
        ax.scatter(x, y, s=4, alpha=0.2, color="0.5", linewidths=0)
        edges = np.unique(np.round(np.geomspace(x.min(), x.max() + 1, 9)).astype(int))
        cx, cy = [], []
        binid = np.digitize(x, edges[1:-1])
        for b in range(len(edges) - 1):
            m = binid == b
            if m.sum():
                cx.append(np.sqrt(edges[b] * edges[b + 1])); cy.append(np.median(y[m]))
        ax.plot(cx, cy, "o-", color="crimson", lw=2, ms=6, label="binned median")
        rho = _safe_spearman(np.log10(x), y)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_atoms (training-coverage proxy, log)")
        ax.set_ylabel(f"{ycol} (log)")
        ax.set_title(f"{lab} force error vs frequency   (Spearman log n vs err = {rho:+.3f})")
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"  wrote plot -> {plot_out}")


try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


# ----------------------------------------------------------------------------
# Preprocessing: per-feature z-score, then PCA via the DxD covariance matrix
# ----------------------------------------------------------------------------
def standardize_and_pca(X, var_keep=0.95, max_dims=50, cosine=False, chunk=200_000):
    n, d = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0

    # Accumulate the DxD covariance of the standardized data in chunks.
    C = np.zeros((d, d), dtype=np.float64)
    for s in range(0, n, chunk):
        Z = (X[s:s + chunk] - mean) / std
        C += Z.T @ Z
    C /= max(n - 1, 1)

    evals, evecs = np.linalg.eigh(C)          # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals[evals < 0] = 0.0
    cumvar = np.cumsum(evals) / max(evals.sum(), 1e-12)
    k = int(np.searchsorted(cumvar, var_keep) + 1)
    k = max(1, min(k, max_dims, d))
    V = evecs[:, :k].astype(np.float32)

    Xp = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        Z = ((X[s:s + chunk] - mean) / std).astype(np.float32)
        Xp[s:s + chunk] = Z @ V
    if cosine:                                 # cosine geometry: project onto unit sphere
        nrm = np.linalg.norm(Xp, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        Xp /= nrm
    print(f"  PCA: {d} -> {k} dims ({cumvar[k-1]*100:.1f}% var)"
          f"{' + L2-normalised (cosine)' if cosine else ''}", flush=True)
    return Xp


# ----------------------------------------------------------------------------
# Exact kNN. Both backends return EXACT Euclidean neighbours, so results are
# identical (modulo tie-breaks at equal distance); only the speed differs.
#   ckdtree : scipy cKDTree, multithreaded (workers=-1). No extra dependency.
#   faiss   : faiss IndexFlatL2 (brute force, exact). 10-100x faster at scale;
#             '--knn-backend faiss-gpu' offloads the search to the GPU.
# We do NOT use faiss approximate indexes (IVF/HNSW) anywhere: those would make
# the neighbours -- and hence purity -- approximate.
# ----------------------------------------------------------------------------
def knn_indices(Xp, k, backend="auto"):
    """Return (n, k) array of nearest-neighbour indices, self excluded."""
    if backend == "auto":
        backend = "faiss" if _HAVE_FAISS else "ckdtree"
    if backend in ("faiss", "faiss-gpu"):
        if not _HAVE_FAISS:
            raise SystemExit("--knn-backend faiss requested but faiss not importable "
                             "(install faiss-cpu or faiss-gpu).")
        X = np.ascontiguousarray(Xp, dtype=np.float32)
        index = faiss.IndexFlatL2(X.shape[1])          # exact L2
        if backend == "faiss-gpu":
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(X)
        _, idx = index.search(X, k + 1)                 # +1: first hit is self
    else:
        tree = cKDTree(Xp)
        _, idx = tree.query(Xp, k=k + 1, workers=-1)
    return idx[:, 1:]                                   # drop self column


# ----------------------------------------------------------------------------
# Metric 1: global kNN label purity.
# ----------------------------------------------------------------------------
def knn_purity(Xp, labels_int, k, backend="auto"):
    neigh = knn_indices(Xp, k, backend=backend)
    same = (labels_int[neigh] == labels_int[:, None])
    return same.mean(axis=1)                   # per-atom purity in [0,1]


# ----------------------------------------------------------------------------
# Metric 2: size-matched compactness null. The null distribution of a class's
# mean-squared-radius depends only on its size n, so we bootstrap r2 for a grid
# of sizes ONCE and interpolate, instead of bootstrapping every class.
#
# Cost control at full-dataset scale (these keep results equivalent to the naive
# full bootstrap to within Monte-Carlo noise):
#   * grid TOPS OUT at the largest class size, not N -- the null is only ever
#     queried at individual class sizes, so grid points larger than any class
#     were pure waste (and the most expensive ones).
#   * per draw we sample WITH replacement (a true bootstrap) -- removes the
#     O(N) permutation that rng.choice(replace=False) does for large sizes.
#   * each draw gathers at most `cap` rows. For sizes n <= cap we bootstrap
#     directly (identical to before, since cap << N => negligible collisions).
#     For n > cap we use the analytic mean  E[r2] = sigma2_global*(n-1)/n  and
#     scale the std by sqrt(cap/n) -- the mean-of-n estimator's std ~ 1/sqrt(n)
#     (CLT) -- so we never gather more than cap rows regardless of class size.
# ----------------------------------------------------------------------------
def compactness_null(Xp, sigma2_pool, max_size, n_boot=150, cap=100_000, seed=0,
                     pool_idx=None):
    """Bootstrap RMS-radius^2 null vs size on a log grid up to max_size.

    Returns (logg, g_mean, g_std): plain arrays (picklable, unlike a closure) to
    be fed to np.interp(log n, ...) by each worker.

    pool_idx: if given, draws the null subsets ONLY from these rows of Xp (an
    element-conditioned null), and `sigma2_pool` must be that pool's dispersion;
    if None, draws from all of Xp (the global null) and sigma2_pool == sigma2_global.
    """
    rng = np.random.default_rng(seed)
    N = len(Xp) if pool_idx is None else len(pool_idx)
    top = max(3, min(int(max_size), N))
    grid = np.unique(np.clip(
        np.round(np.geomspace(2, top, 35)).astype(int), 2, top))
    g_mean, g_std = [], []
    for n in grid:
        m = int(min(n, cap))
        r2 = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            loc = rng.integers(0, N, size=m)           # with replacement (bootstrap)
            sel = loc if pool_idx is None else pool_idx[loc]
            S = Xp[sel]
            mu = S.mean(axis=0)
            r2[b] = ((S - mu) ** 2).sum(axis=1).mean()
        if n <= cap:
            g_mean.append(r2.mean())
            g_std.append(r2.std() + 1e-12)
        else:                                          # extrapolate beyond the cap
            g_mean.append(sigma2_pool * (n - 1) / n)
            g_std.append(r2.std() * np.sqrt(m / n) + 1e-12)
    return np.log(grid), np.asarray(g_mean), np.asarray(g_std)


def element_null(name, pool_idx, Xp, max_size, n_boot=150, cap=100_000, seed=0):
    """Build the size-matched null for ONE central element: estimate the element
    pool's dispersion (capped subsample for memory), then bootstrap r2 vs size
    drawing only from that element's rows. Top-level/picklable for joblib."""
    rng = np.random.default_rng(seed)
    sub = pool_idx if len(pool_idx) <= cap else pool_idx[rng.integers(0, len(pool_idx), cap)]
    S = Xp[sub]
    mu = S.mean(axis=0)
    sig2 = float(((S - mu) ** 2).sum(axis=1).mean())
    logg, mean, std = compactness_null(Xp, sig2, max_size, n_boot=n_boot, cap=cap,
                                       seed=seed, pool_idx=pool_idx)
    return name, sig2, logg, mean, std


# ----------------------------------------------------------------------------
# Metric 3: connected components of the SYMMETRIC (union) kNN graph within one
# class. Edge i-j iff j is among i's k nearest OR i among j's -> A | A^T.
# We deliberately do NOT use the mutual-kNN graph (A & A^T): mutual-kNN is so
# sparse that even a single Gaussian blob shatters into hundreds of tiny pieces
# The union kNN graph is the standard connectivity graph (spectral clustering,
# Isomap) and a robust scalable proxy for single-linkage / 0-dim persistent
# homology: 1 big component => one connected region; >=2 components each holding
# a real fraction of the points => genuinely disjoint regions in feature space.
# n_components_massive (components with >= massive_frac of the points) is the
# noise-robust "number of modes" -- raw n_components can still pick up a few
# outlier singletons at low density.
# ----------------------------------------------------------------------------
def knn_graph_components(Xc, k_graph, massive_frac=0.10):
    n = len(Xc)
    kk = min(k_graph, n - 1)
    tree = cKDTree(Xc)
    # workers=1: this runs INSIDE a joblib worker (parallelism is across classes),
    # so threading here on top would oversubscribe (n_jobs x ncores). The per-class
    # graph is capped at --graph-cap points, so single-threaded is plenty.
    _, idx = tree.query(Xc, k=kk + 1, workers=1)
    idx = idx[:, 1:]                                   # drop self
    rows = np.repeat(np.arange(n), kk)
    cols = idx.reshape(-1)
    A = csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n))
    M = A.maximum(A.T)                                 # symmetric union
    ncomp, lab = connected_components(M, directed=False)
    sizes = np.bincount(lab)
    largest = sizes.max() / n
    n_massive = int((sizes >= massive_frac * n).sum())
    return ncomp, float(largest), n_massive


# ----------------------------------------------------------------------------
# Per-class expensive metrics (compactness + components). Top-level (picklable)
# so joblib can fan classes out across workers. RESULT-INVARIANT to worker count:
# the only randomness -- subsampling classes above graph_cap -- is drawn from an
# RNG seeded by the class CODE (base_seed + code), not by loop/dispatch order, so
# every class gets the same subsample whether run serial or on N cores.
# Xp is passed once; joblib memmaps it read-only and shares it across workers
# ----------------------------------------------------------------------------
def class_expensive(code, idx, Xp, sigma2_global, null_logg, null_mean, null_std,
                    k_graph, graph_cap, massive_frac, base_seed, err=None, rel=None,
                    fmag=None, e_logg=None, e_mean=None, e_std=None, e_sig2=None):
    Xc_full = Xp[idx]
    n = len(idx)
    mu = Xc_full.mean(axis=0)
    sq = ((Xc_full - mu) ** 2).sum(axis=1)               # per-atom squared radius
    r2 = sq.mean()
    ln = np.log(n)
    nm = float(np.interp(ln, null_logg, null_mean))
    ns = float(np.interp(ln, null_logg, null_std))
    # Element-conditioned compactness: same r2, but vs a null drawn from same-element
    # atoms. >0 => the environment localizes the atom BEYOND its element identity.
    if e_mean is not None:
        enm = float(np.interp(ln, e_logg, e_mean))
        ens = float(np.interp(ln, e_logg, e_std))
        compactness_z_element = float((enm - r2) / ens)
        dispersion_ratio_element = float(np.sqrt(r2 / e_sig2)) if e_sig2 and e_sig2 > 0 \
            else float("nan")
    else:
        compactness_z_element = float("nan")
        dispersion_ratio_element = float("nan")
    rng = np.random.default_rng(base_seed + int(code))   # per-class, order-independent
    Xc = Xc_full
    if n > graph_cap:
        Xc = Xc_full[rng.choice(n, graph_cap, replace=False)]
    ncomp, largest, nmass = knn_graph_components(Xc, k_graph, massive_frac)
    # Intra-token core-vs-periphery: within THIS token (chemistry/element/CN fixed),
    # does an atom's distance from the token centroid track its force error?
    dist = np.sqrt(sq)
    dist_err = _safe_spearman(dist, err[idx]) if err is not None else float("nan")
    dist_relerr = _safe_spearman(dist, rel[idx]) if rel is not None else float("nan")
    # Magnitude diagnostic: within THIS token, does feature-space position track
    # the FORCE MAGNITUDE |F_ref|? A strong +ve rho here (with dist_err +ve and
    # dist_relerr -ve) is the fingerprint that the apparent geometry<->error signal
    # is really geometry<->|F|, not geometry<->accuracy.
    dist_fmag = _safe_spearman(dist, fmag[idx]) if fmag is not None else float("nan")
    return code, dict(
        dispersion_ratio=float(np.sqrt(r2 / sigma2_global)),
        compactness_z=float((nm - r2) / ns),             # +ve => tighter than chance
        compactness_z_element=compactness_z_element,      # vs same-element null
        dispersion_ratio_element=dispersion_ratio_element,
        n_components=ncomp,
        largest_comp_frac=largest,
        n_components_massive=nmass,
        dist_err_spearman=dist_err,
        dist_relerr_spearman=dist_relerr,
        dist_fmag_spearman=dist_fmag,
    )


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def load_data(feature_glob, meta_glob, max_atoms, seed=0):
    fpaths = sorted(glob.glob(feature_glob))
    mpaths = sorted(glob.glob(meta_glob))
    if not fpaths or len(fpaths) != len(mpaths):
        raise SystemExit(f"feature/meta shard mismatch: {len(fpaths)} vs {len(mpaths)}")
    metas, shard_of, sizes = [], [], []
    for si, mp in enumerate(mpaths):
        m = pd.read_parquet(mp)
        m["shard"] = si
        metas.append(m)
        sizes.append(len(m))
    meta = pd.concat(metas, ignore_index=True)
    print(f"  loaded {len(meta):,} atoms across {len(fpaths)} shards", flush=True)

    sel = np.arange(len(meta))
    if max_atoms and len(meta) > max_atoms:    # stratified subsample by chemenv_id
        rng = np.random.default_rng(seed)
        frac = max_atoms / len(meta)
        keep = []
        for _, grp in meta.groupby("chemenv_id", sort=False):
            g = grp.index.to_numpy()
            ntake = max(1, int(round(len(g) * frac)))
            keep.append(rng.choice(g, min(ntake, len(g)), replace=False))
        sel = np.sort(np.concatenate(keep))
        print(f"  stratified subsample -> {len(sel):,} atoms", flush=True)

    # Gather only the selected feature rows from each shard memmap.
    D = np.load(fpaths[0], mmap_mode="r").shape[1]
    X = np.empty((len(sel), D), dtype=np.float32)
    meta_sel = meta.iloc[sel].reset_index(drop=True)
    out = 0
    for si, fp in enumerate(fpaths):
        mm = np.load(fp, mmap_mode="r")
        local = meta_sel.index[meta_sel["shard"] == si].to_numpy()
        if len(local) == 0:
            continue
        rows = meta_sel.loc[local, "row"].to_numpy()
        X[local] = mm[rows]
        out += len(local)
    assert out == len(sel)
    return X, meta_sel


def make_selftest():
    """Fabricate labelled blobs with known structure to validate the metrics."""
    rng = np.random.default_rng(42)
    D = 64
    parts, labels = [], []

    def blob(center, n, scale):
        return center + scale * rng.standard_normal((n, D))

    c = rng.standard_normal((6, D)) * 8.0
    # tight unimodal
    parts.append(blob(c[0], 4000, 0.4)); labels += ["TIGHT_UNI"] * 4000
    # bimodal: same label, two far-apart blobs -> expect ~2 components, lower purity
    parts.append(blob(c[1], 2000, 0.4)); labels += ["BIMODAL"] * 2000
    parts.append(blob(c[1] + c[2] * 0.9, 2000, 0.4)); labels += ["BIMODAL"] * 2000
    # diffuse: spread across the whole space -> low purity, compactness_z ~ 0
    parts.append(blob(np.zeros(D), 3000, 6.0)); labels += ["DIFFUSE"] * 3000
    # small tight
    parts.append(blob(c[3], 200, 0.3)); labels += ["SMALL_TIGHT"] * 200
    X = np.vstack(parts).astype(np.float32)
    # Two central elements, each spanning >1 token, so the element-conditioned null is
    # meaningful: A={TIGHT_UNI, BIMODAL}, B={DIFFUSE, SMALL_TIGHT}. A tight token should
    # be tighter than its same-element pool (compactness_z_element > 0).
    lab2elem = {"TIGHT_UNI": "A", "BIMODAL": "A", "DIFFUSE": "B", "SMALL_TIGHT": "B"}
    meta = pd.DataFrame({
        "chemenv_id": labels,
        "f_ae_mace": rng.random(len(labels)).astype(np.float32),
        "ref_fmag": (rng.random(len(labels)) + 0.5).astype(np.float32),  # exercise rel-err path
        "element": [lab2elem[l] for l in labels],
        "sid": ["s"] * len(labels),
        "row": np.arange(len(labels)),
        "shard": 0,
    })
    return X, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-glob")
    ap.add_argument("--meta-glob")
    ap.add_argument("--out", default="chemenv_feature_report.parquet")
    ap.add_argument("--k", type=int, default=30, help="k for purity kNN.")
    ap.add_argument("--k-graph", type=int, default=15, help="k for the kNN graph.")
    ap.add_argument("--var-keep", type=float, default=0.95)
    ap.add_argument("--max-dims", type=int, default=50)
    ap.add_argument("--cosine", action="store_true", help="Use cosine geometry.")
    ap.add_argument("--max-atoms", type=int, default=1_500_000)
    ap.add_argument("--min-size", type=int, default=20,
                    help="Min atoms in a chemenv_id to compute components/compactness.")
    ap.add_argument("--graph-cap", type=int, default=20_000,
                    help="Subsample classes above this for the component graph.")
    ap.add_argument("--null-cap", type=int, default=100_000,
                    help="Max rows gathered per compactness-null bootstrap draw; "
                         "sizes above this use the analytic mean + sqrt(cap/n) std.")
    ap.add_argument("--null-max-size", type=int, default=0,
                    help="Top of the compactness-null size grid (0 = largest class).")
    ap.add_argument("--no-element-null", dest="element_null", action="store_false",
                    help="Skip the per-central-element compactness null "
                         "(compactness_z_element); keeps only the global null.")
    ap.add_argument("--massive-frac", type=float, default=0.10,
                    help="A component counts as 'massive' if it holds >= this fraction.")
    ap.add_argument("--knn-backend", default="auto",
                    choices=["auto", "faiss", "faiss-gpu", "ckdtree"],
                    help="Exact-kNN engine for purity. auto=faiss if installed else cKDTree.")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="Worker processes for the per-class metrics (-1 = all cores).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed; per-class subsample uses seed+class_code (order-independent).")
    ap.add_argument("--summary-rows", type=int, default=0,
                    help="Rows to print in the console summary table (0 = all ids "
                         ">= min-size; the parquet always holds every id).")
    ap.add_argument("--no-plot", dest="plot", action="store_false",
                    help="Skip the compactness-vs-error scatter plot.")
    ap.add_argument("--plot-out", default=None,
                    help="PNG path for the compactness-vs-error scatter "
                         "(default: alongside --out).")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if args.selftest:
        print("SELF-TEST: fabricating labelled blobs ...", flush=True)
        X, meta = make_selftest()
    else:
        if not (args.feature_glob and args.meta_glob):
            raise SystemExit("--feature-glob and --meta-glob required (or use --selftest)")
        print("Loading features + metadata ...", flush=True)
        X, meta = load_data(args.feature_glob, args.meta_glob, args.max_atoms)

    # Drop INVALID env ids from the analysis (keep nothing downstream depends on).
    valid = ~meta["chemenv_id"].str.contains("INVALID", na=False)
    if (~valid).any():
        print(f"  dropping {(~valid).sum():,} INVALID-env atoms", flush=True)
        X, meta = X[valid.to_numpy()], meta[valid].reset_index(drop=True)

    print("Preprocessing (standardize + PCA) ...", flush=True)
    Xp = standardize_and_pca(X, var_keep=args.var_keep, max_dims=args.max_dims,
                             cosine=args.cosine)

    # Integer-code labels for fast grouping.
    cats = meta["chemenv_id"].astype("category")
    labels_int = cats.cat.codes.to_numpy()
    classes = list(cats.cat.categories)
    n_classes = len(classes)

    # Per-atom errors. f_ae_mace is |F_mace - F_ref| in meV/A; ref_fmag is |F_ref|
    # in meV/A (added by extract_mace_features.py), so rel_err is dimensionless.
    err_atom = meta["f_ae_mace"].to_numpy().astype(np.float64)
    if "ref_fmag" in meta.columns:
        fmag_atom = meta["ref_fmag"].to_numpy().astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_atom = np.where(fmag_atom > 0, err_atom / fmag_atom, np.nan)
        print(f"  relative force error available "
              f"(ref_fmag present; {np.isfinite(rel_atom).sum():,} finite)", flush=True)
    else:
        rel_atom = None
        fmag_atom = None
        print("  ref_fmag absent -> relative-error + magnitude-diagnostic columns "
              "skipped (re-run extract_mace_features.py to add them)", flush=True)

    backend = "faiss" if (args.knn_backend == "auto" and _HAVE_FAISS) else \
              ("ckdtree" if args.knn_backend == "auto" else args.knn_backend)
    print(f"Metric 1: kNN purity (k={args.k}, backend={backend}) ...", flush=True)
    t = time.time()
    purity_atom = knn_purity(Xp, labels_int, args.k, backend=args.knn_backend)
    print(f"  done in {time.time()-t:.1f}s", flush=True)

    print("Metric 2: compactness null model ...", flush=True)
    t = time.time()
    sigma2_global = ((Xp - Xp.mean(axis=0)) ** 2).sum(axis=1).mean()
    class_counts = np.bincount(labels_int, minlength=n_classes)
    max_size = int(args.null_max_size) if args.null_max_size > 0 else int(class_counts.max())
    null_logg, null_mean, null_std = compactness_null(
        Xp, sigma2_global, max_size, cap=args.null_cap, seed=args.seed)
    print(f"  null built in {time.time()-t:.1f}s "
          f"(grid up to n={max_size:,}, per-draw cap {args.null_cap:,}; "
          f"global RMS radius {np.sqrt(sigma2_global):.3f})", flush=True)

    # Per-central-element null: same machinery, but each element's null is drawn only
    # element identity (factoring out the dominant element axis of the latent space).
    enull = {}                                   # element name -> (logg, mean, std, sig2)
    if args.element_null:
        elem_atom = meta["element"].astype("category")
        elem_codes = elem_atom.cat.codes.to_numpy()
        elem_names = list(elem_atom.cat.categories)
        elem_of_class = meta.groupby(labels_int)["element"].first().to_dict()
        # cover only elements that own a >= min-size token; grid each up to that
        # element's largest token (queried sizes never exceed it).
        need = sorted({elem_of_class[c] for c in range(n_classes)
                       if class_counts[c] >= args.min_size})
        elem_max = {}
        for c in range(n_classes):
            e = elem_of_class[c]
            elem_max[e] = max(elem_max.get(e, 0), int(class_counts[c]))
        print(f"Metric 2b: per-element compactness null for {len(need)} elements ...",
              flush=True)
        t = time.time()
        name_to_ecode = {nm: i for i, nm in enumerate(elem_names)}
        pools = {nm: np.where(elem_codes == name_to_ecode[nm])[0] for nm in need}
        e_results = Parallel(n_jobs=args.n_jobs)(
            delayed(element_null)(nm, pools[nm], Xp, elem_max[nm],
                                  cap=args.null_cap, seed=args.seed)
            for nm in need)
        for nm, sig2, lg, mn, sd in e_results:
            enull[nm] = (lg, mn, sd, sig2)
        print(f"  per-element nulls built in {time.time()-t:.1f}s", flush=True)
    else:
        elem_of_class = {}

    # Cheap per-class fields (size, element, purity mean, force errors) -- all
    # vectorised groupby, no Python loop.
    work = pd.DataFrame({"code": labels_int, "purity": purity_atom,
                         "f_ae_mace": meta["f_ae_mace"].to_numpy(),
                         "element": meta["element"].to_numpy()})
    if rel_atom is not None:
        work["rel_err"] = rel_atom
    if fmag_atom is not None:
        work["fmag"] = fmag_atom
    g = work.groupby("code", sort=True)
    base = pd.DataFrame({
        "chemenv_id": [classes[c] for c in g.size().index],
        "element": g["element"].first().to_numpy(),
        "n_atoms": g.size().to_numpy().astype(np.int64),
        "knn_purity": g["purity"].mean().to_numpy(),
        "mean_force_error": g["f_ae_mace"].mean().to_numpy(),
        "median_force_error": g["f_ae_mace"].median().to_numpy(),
    }, index=g.size().index)
    if rel_atom is not None:
        base["mean_rel_force_error"] = g["rel_err"].mean().to_numpy()
        base["median_rel_force_error"] = g["rel_err"].median().to_numpy()
    if fmag_atom is not None:
        base["mean_fmag"] = g["fmag"].mean().to_numpy()       # |F_ref| meV/A per token

    # Per-class row indices (one argsort + slicing, not C scans of the array).
    order = np.argsort(labels_int, kind="stable")
    sorted_lab = labels_int[order]
    starts = np.searchsorted(sorted_lab, np.arange(n_classes))
    ends = np.searchsorted(sorted_lab, np.arange(n_classes) + 1)
    idx_per_class = {c: order[starts[c]:ends[c]] for c in range(n_classes)}

    codes_big = [c for c in range(n_classes)
                 if (ends[c] - starts[c]) >= args.min_size]
    print(f"Metric 3: kNN-graph components (k={args.k_graph}) for "
          f"{len(codes_big)}/{n_classes} ids on n_jobs={args.n_jobs} ...", flush=True)
    t = time.time()
    # joblib memmaps Xp once and shares it read-only across workers; per-class
    # subsampling is seeded by class code -> identical output at any --n-jobs.
    def _enull(c):                                   # (logg, mean, std, sig2) for c's element
        return enull.get(elem_of_class.get(c), (None, None, None, None))
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(class_expensive)(
            c, idx_per_class[c], Xp, sigma2_global, null_logg, null_mean, null_std,
            args.k_graph, args.graph_cap, args.massive_frac, args.seed,
            err_atom, rel_atom, fmag_atom, *_enull(c))
        for c in codes_big)
    print(f"  done in {(time.time()-t)/60:.2f} min", flush=True)

    exp_cols = ["dispersion_ratio", "compactness_z", "compactness_z_element",
                "dispersion_ratio_element", "n_components",
                "largest_comp_frac", "n_components_massive",
                "dist_err_spearman", "dist_relerr_spearman", "dist_fmag_spearman"]
    for col in exp_cols:
        base[col] = np.nan
    for code, rec in results:
        for col in exp_cols:
            base.at[code, col] = rec[col]

    report = base.reset_index(drop=True).sort_values("n_atoms", ascending=False)
    report.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out} ({len(report)} chemenv_ids) in "
          f"{(time.time()-t0)/60:.2f} min", flush=True)

    # Quick console summary.
    big = report[report["n_atoms"] >= args.min_size]
    print("\n=== summary (chemenv_ids with >= min-size atoms) ===")
    cols = ["chemenv_id", "n_atoms", "knn_purity", "compactness_z",
            "compactness_z_element", "n_components", "largest_comp_frac",
            "mean_force_error"]
    if "mean_rel_force_error" in big.columns:
        cols += ["mean_rel_force_error"]
    cols += ["dist_err_spearman"]
    table = big[cols] if args.summary_rows <= 0 else big[cols].head(args.summary_rows)
    print(f"({len(table)} of {len(big)} ids shown; full set in {args.out})")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(table.to_string(index=False))
    multimodal = big[(big["n_components_massive"] >= 2) & (big["largest_comp_frac"] < 0.9)]
    print(f"\nFlagged multimodal (>=2 massive components, largest <90%): "
          f"{len(multimodal)}/{len(big)} ids -> candidates for the "
          f"HDBSCAN / persistent-homology pass")

    # Between-token correlations: each metric vs absolute (and relative) force error.
    err_targets = ["mean_force_error"]
    if "mean_rel_force_error" in big.columns:
        err_targets.append("mean_rel_force_error")
    for tgt in err_targets:
        if big[tgt].notna().sum() > 3:
            print(f"  -- between-token corr vs {tgt}:")
            for m in ["knn_purity", "compactness_z", "compactness_z_element",
                      "largest_comp_frac"]:
                sub = big[[m, tgt]].dropna()
                if len(sub) > 3:
                    r = np.corrcoef(sub[m], sub[tgt])[0, 1]
                    print(f"     corr({m}, {tgt}) = {r:+.3f}")

    # MAGNITUDE DIAGNOSTIC (between-token): is the geometry<->error signal really
    # geometry<->|F_ref|? Correlate each geometry metric (and both error definitions)
    # against the token's mean force magnitude. Expected confound signature:
    #   compactness_z, knn_purity vs mean_fmag : negative (diffuse/impure = larger |F|)
    #   mean_force_error          vs mean_fmag : strongly positive (abs err scales with |F|)
    #   mean_rel_force_error      vs mean_fmag : ~0 / negative (magnitude divided out)
    # If these hold, the apparent compactness<->abs-error link is force magnitude, not accuracy.
    if "mean_fmag" in big.columns and big["mean_fmag"].notna().sum() > 3:
        print("  -- MAGNITUDE DIAGNOSTIC: between-token corr vs mean_fmag (|F_ref|):")
        mag_targets = ["compactness_z", "compactness_z_element", "knn_purity",
                       "largest_comp_frac", "mean_force_error"]
        if "mean_rel_force_error" in big.columns:
            mag_targets.append("mean_rel_force_error")
        for m in mag_targets:
            r, nn = _pearson(big, m, "mean_fmag")
            print(f"     corr({m}, mean_fmag) = {r:+.3f}  (n={nn})")

    # Intra-token (core-vs-periphery): distribution of per-token Spearman(distance,error).
    # This holds element/coordination/chemistry fixed, so it isolates feature-space
    # position -> error without the between-token confounds.
    print("  -- intra-token distance->error (per-token Spearman):")
    for col, lab in [("dist_err_spearman", "abs err"),
                     ("dist_relerr_spearman", "rel err"),
                     ("dist_fmag_spearman", "|F_ref|  <- magnitude diagnostic")]:
        if col in big.columns:
            s = big[col].dropna()
            if len(s) > 3:
                print(f"     {lab}: median rho={s.median():+.3f}, "
                      f"frac rho>0 = {100*(s > 0).mean():.0f}%  ({len(s)} tokens)")
    print("  (NOTE: control for central element + force magnitude before "
          "reading the between-token numbers as causal -- per your earlier finding.)")

    # compactness_z vs error under confound filters (CN-based + z-floor) + scatter plot.
    plot_out = args.plot_out or (args.out.rsplit(".", 1)[0] + "_compactness_vs_error.png")
    compactness_error_report(big, plot=args.plot, plot_out=plot_out)

    # Learning-curve test: frequency (n_atoms) vs force error, binned + Spearman + plot.
    freq_plot_out = args.out.rsplit(".", 1)[0] + "_frequency_vs_error.png"
    frequency_error_report(big, plot=args.plot, plot_out=freq_plot_out)


if __name__ == "__main__":
    main()
