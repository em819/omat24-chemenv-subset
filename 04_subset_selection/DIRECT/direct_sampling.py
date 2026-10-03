#!/usr/bin/env python3
# As-run filename: direct_sampling_standalone.py
"""DIRECT steps 3-5: cached M3GNet features -> selected subset.

DIRECT = DImensionality-Reduced Encoded Clusters with sTratified sampling,
Qi, Ko, Wood, Pham, Ong, npj Comput. Mater. 10, 43 (2024), arXiv:2307.13710.

The classes below reproduce materialsvirtuallab/maml
src/maml/sampling/{pca,clustering,stratified_sampling,direct}.py with numpy and
scikit-learn only (no matgl / maml needed):
    PrincipalComponentAnalysis  Kaiser's rule + explained-variance weighting
    BirchClustering             threshold shrink loop to reach n clusters
    SelectKFromClusters         distance-to-centroid ranking, k at even intervals
    DIRECTSampler               StandardScaler -> PCA -> BIRCH -> SelectK

The only departure from maml is the starting BIRCH radius in --target-size mode:
_auto_threshold finds it by bisection on the full data instead of starting from
maml's 0.5. The result equals maml's DIRECTSampler with threshold_init set to
the value printed in the log.

Paper run (--target-size mode):
    python direct_sampling.py --features feats.npy --ids feats.ids.json \\
           --n-clusters 116500 --target-size 460000 --out selected_460k
    -> 21 PCs (81.96 % variance), threshold 0.1092, 116,328 clusters, k = 4,
       458,754 structures

Checks:
    python direct_sampling.py --self-test        # step-5 protocol vs a brute-force reference
    python direct_sampling.py --verify-vs-maml   # identical selection to maml (needs `pip install maml`)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =========================================================================== #
#  DIRECT core (steps 3-5), mirrors maml                                       #
# =========================================================================== #
class PrincipalComponentAnalysis(BaseEstimator, TransformerMixin):
    """Keep the PCs with eigenvalue > 1 (Kaiser's rule), optionally weighted by
    their explained-variance ratio. Mirrors maml.sampling.pca."""

    def __init__(self, weighting_PCs: bool = True):
        self.pca = PCA()
        self.weighting_PCs = weighting_PCs

    def fit(self, normalized_features, y=None):
        self.pca.fit(normalized_features)
        return self

    def transform(self, normalized_features):
        m = len([e for e in self.pca.explained_variance_ if e > 1])
        explained_variance = self.pca.explained_variance_ratio_
        logger.info(
            f"Selected first {m} PCs, explaining "
            f"{100 * sum(explained_variance[:m]):.2f}% variance"
        )
        if not self.weighting_PCs:
            return self.pca.transform(normalized_features)[:, :m]
        return self.pca.transform(normalized_features)[:, :m] * explained_variance[:m]


class BirchClustering(BaseEstimator, TransformerMixin):
    """BIRCH into n clusters; if the CF-tree has fewer than n leaf subclusters,
    shrink the radius threshold and refit. Mirrors maml.sampling.clustering."""

    def __init__(self, n=None, threshold_init: float = 0.5, **kwargs):
        self.n = n
        self.threshold_init = threshold_init
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, PCAfeatures):
        model = Birch(n_clusters=self.n, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)
        if self.n is not None:
            while len(set(model.subcluster_labels_)) < self.n:
                self.threshold_init = self.threshold_init / self.n * len(set(model.subcluster_labels_))
                model = Birch(n_clusters=self.n, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)

        labels = model.predict(PCAfeatures)
        self.model = model
        label_centers = dict(zip(model.subcluster_labels_, model.subcluster_centers_))
        return {
            "labels": labels,
            "label_centers": label_centers,
            "PCAfeatures": PCAfeatures,
        }


class SelectKFromClusters(BaseEstimator, TransformerMixin):
    """Pick k structures per cluster: sort by distance to the centroid and take k
    at constant index intervals (all of the cluster if it has <= k members).
    Mirrors maml.sampling.stratified_sampling.

    selection_criteria: "center" (paper), "smallest" (fewest atoms; needs
    n_sites) or "random". allow_duplicate=False collapses the repeated indices
    that linspace produces when k > cluster size.
    """

    def __init__(self, k: int = 1, allow_duplicate: bool = False,
                 selection_criteria: str = "center", n_sites=None):
        self.k = k
        self.allow_duplicate = allow_duplicate
        allowed_selection_criterion = ["random", "smallest", "center"]
        if selection_criteria not in allowed_selection_criterion:
            raise ValueError(f"Invalid selection_criteria, it must be one of {allowed_selection_criterion}.")
        if selection_criteria == "smallest" and not n_sites:
            raise ValueError('n_sites must be provided when selection_criteria="smallest."')
        self.selection_criteria = selection_criteria
        self.n_sites = n_sites

    def fit(self, X, y=None):
        return self

    def transform(self, clustering_data: dict):
        if any(key not in clustering_data for key in ["labels", "PCAfeatures"]):
            raise Exception(
                "The data returned by clustering step should at least provide label and feature information."
            )
        if self.selection_criteria == "center" and "label_centers" not in clustering_data:
            warnings.warn(
                "Centroid location is not provided, so random selection from each cluster will be performed, "
                "which likely will still outperform manual sampling in terms of feature coverage. "
            )
        if self.selection_criteria == "smallest":
            try:
                assert len(self.n_sites) == len(clustering_data["PCAfeatures"])
            except Exception:
                raise ValueError("n_sites must have same length as features processed in clustering.")

        selected_indexes = []
        for label in set(clustering_data["labels"]):
            indexes_same_label = np.where(label == clustering_data["labels"])[0]
            features_same_label = clustering_data["PCAfeatures"][indexes_same_label]
            n_same_label = len(features_same_label)
            if "label_centers" in clustering_data and self.selection_criteria == "center":
                center_same_label = clustering_data["label_centers"][label]
                distance_to_center = np.linalg.norm(
                    features_same_label - center_same_label, axis=1
                ).reshape(len(indexes_same_label))
                select_k_indexes = np.array([int(i) for i in np.linspace(0, n_same_label - 1, self.k)])
                selected_indexes.extend(
                    indexes_same_label[np.argpartition(distance_to_center, select_k_indexes)[select_k_indexes]]
                )
            elif self.selection_criteria == "smallest":
                if self.k >= n_same_label:
                    selected_indexes.extend(indexes_same_label)
                else:
                    select_k_indexes = np.arange(self.k)
                    selected_indexes.extend(
                        indexes_same_label[
                            np.argpartition(
                                np.array(self.n_sites)[indexes_same_label],
                                select_k_indexes,
                            )[select_k_indexes]
                        ]
                    )
            else:
                selected_indexes.extend(indexes_same_label[np.random.randint(n_same_label, size=self.k)])
        n_duplicate = len(selected_indexes) - len(set(selected_indexes))
        if not self.allow_duplicate and n_duplicate > 0:
            selected_indexes = list(set(selected_indexes))
        elif self.allow_duplicate and n_duplicate > 0:
            warnings.warn(f"There are {n_duplicate} duplicated selections.")
        logger.info(f"Finally selected {len(selected_indexes)} configurations.")
        return {
            "PCAfeatures": clustering_data["PCAfeatures"],
            "selected_indexes": selected_indexes,
        }


class DIRECTSampler(Pipeline):
    """maml.sampling.direct.DIRECTSampler without the encoder step: it takes the
    precomputed (N, D) feature array of featurize_m3gnet.py.

        out = DIRECTSampler(n_clusters=..., k=...).fit_transform(features)
        idx = out["selected_indexes"]
    """

    def __init__(
        self,
        scaler="StandardScaler",
        weighting_PCs: bool = True,
        n_clusters=None,
        threshold_init: float = 0.5,
        k: int = 1,
        allow_duplicate: bool = False,
        selection_criteria: str = "center",
        n_sites=None,
        birch_kwargs: dict | None = None,
    ):
        self.scaler = StandardScaler() if scaler == "StandardScaler" else scaler
        self.weighting_PCs = weighting_PCs
        self.n_clusters = n_clusters
        self.threshold_init = threshold_init
        self.k = k
        self.allow_duplicate = allow_duplicate
        self.selection_criteria = selection_criteria
        self.n_sites = n_sites
        self.birch_kwargs = birch_kwargs or {}

        self.pca = PrincipalComponentAnalysis(weighting_PCs=weighting_PCs)
        self.clustering = BirchClustering(
            n=n_clusters, threshold_init=threshold_init, **self.birch_kwargs
        )
        self.select_k_from_clusters = SelectKFromClusters(
            k=k,
            allow_duplicate=allow_duplicate,
            selection_criteria=selection_criteria,
            n_sites=n_sites,
        )
        steps = [
            (i.__class__.__name__, i)
            for i in [self.scaler, self.pca, self.clustering, self.select_k_from_clusters]
            if i is not None and i is not False
        ]
        super().__init__(steps)


# =========================================================================== #
#  Driver                                                                      #
# =========================================================================== #
def run(features: np.ndarray, n_clusters, k, weighting_PCs=True,
        threshold_init=0.5, allow_duplicate=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
    sampler = DIRECTSampler(
        weighting_PCs=weighting_PCs,
        n_clusters=n_clusters,
        threshold_init=threshold_init,
        k=k,
        allow_duplicate=allow_duplicate,
    )
    out = sampler.fit_transform(features)
    return sorted(int(i) for i in out["selected_indexes"]), sampler


# --------------------------------------------------------------------------- #
#  Target-size mode: cluster once at n, then choose k.
#
#  For a fixed clustering the subset size is yield(k) = sum_c min(k, size_c),
#  monotonic in k, so steps 3-4 (with the single-threaded BIRCH) run once and k
#  is solved from the cluster-size histogram.
# --------------------------------------------------------------------------- #
def _auto_threshold(pca_features, n_clusters, cap_factor=1.5, subsample=2_000_000,
                    max_full_probes=8, seed=0):
    """Find a BIRCH radius threshold that gives between n_clusters and
    cap_factor * n_clusters leaf subclusters on the FULL feature cloud.

    BIRCH's threshold is an absolute radius, and with explained-variance
    weighting the cloud is small compared with maml's default 0.5. maml's shrink
    loop then cuts the threshold in one step by (subclusters found / n), which
    can give far more than n subclusters, and the O(m^2) Ward global step on m
    subclusters runs out of memory. The subcluster count at a fixed threshold
    also grows with the number of points, so a subsample calibration does not
    transfer to the full data.

    Probes use Birch(n_clusters=None), which builds only the CF-tree. A
    subsample table gives the first guess, one full probe gives the
    full/subsample ratio, then bracketing and bisection on the full data.
    Only the parameter value is chosen here; the DIRECT classes are unchanged.
    """
    n = int(n_clusters)
    cap = int(cap_factor * n)
    mid_target = (n + cap) // 2
    N = pca_features.shape[0]
    rng = np.random.RandomState(seed)

    def count_full(thr):
        return int(Birch(n_clusters=None, threshold=thr).fit(pca_features).subcluster_centers_.shape[0])

    # Subsample table: threshold <-> number of subclusters.
    if N > subsample:
        Xs = np.ascontiguousarray(pca_features[np.sort(rng.choice(N, subsample, replace=False))])
    else:
        Xs = np.ascontiguousarray(pca_features)
    R = float(np.linalg.norm(Xs - Xs.mean(axis=0), axis=1).max())
    if not np.isfinite(R) or R <= 0:
        logger.warning("[auto-threshold] degenerate feature scale (R<=0); falling back to 0.5")
        return 0.5
    ts, ms, t = [], [], R
    for _ in range(40):
        m = int(Birch(n_clusters=None, threshold=t).fit(Xs).subcluster_centers_.shape[0])
        ts.append(t); ms.append(m)
        if m > 3 * cap:
            break
        t *= 0.75
    ts, ms = np.asarray(ts, float), np.asarray(ms, float)
    order = np.argsort(ms)
    lm, lt = np.log(ms[order]), np.log(ts[order])

    def sub_thr_for(m_target):
        m_target = min(max(m_target, ms.min()), ms.max())
        return float(np.exp(np.interp(np.log(m_target), lm, lt)))

    del Xs

    # Full-data search.
    best_ge_n = None            # (thr, m) with m >= n and the smallest such m
    lo = hi = None              # lo: m > cap (small thr); hi: m < n (large thr)

    def probe(thr):
        nonlocal best_ge_n, lo, hi
        m = count_full(thr)
        logger.info(f"[auto-threshold] full: threshold={thr:.4g} -> {m} subclusters (target [{n}, {cap}])")
        if m >= n and (best_ge_n is None or m < best_ge_n[1]):
            best_ge_n = (thr, m)
        if m > cap:
            lo = thr
        elif m < n:
            hi = thr
        return m

    thr = sub_thr_for(mid_target)
    m = probe(thr)
    if n <= m <= cap:
        return thr
    # Correct by the full/subsample ratio at this threshold and re-aim.
    m_sub_here = float(np.exp(np.interp(np.log(thr), np.log(ts[np.argsort(ts)]),
                                        np.log(ms[np.argsort(ts)]))))
    ratio = max(m / max(m_sub_here, 1.0), 1e-6)
    thr = sub_thr_for(mid_target / ratio)
    m = probe(thr)
    if n <= m <= cap:
        return thr

    # Bracket by a geometric walk, then bisect.
    step = 1.5 if m > cap else (1.0 / 1.5)
    for _ in range(max_full_probes):
        if lo is not None and hi is not None:
            break
        thr *= step
        m = probe(thr)
        if n <= m <= cap:
            return thr
        step = 1.5 if m > cap else (1.0 / 1.5)
    if lo is not None and hi is not None:
        for _ in range(max_full_probes):
            mid = (lo * hi) ** 0.5
            m = probe(mid)
            if n <= m <= cap:
                return mid

    # Fallback: the largest threshold that still gives >= n subclusters.
    if best_ge_n is not None:
        if best_ge_n[1] > cap:
            logger.warning(f"[auto-threshold] could not get subcluster count <= {cap} while >= {n}; "
                           f"using threshold={best_ge_n[0]:.4g} (m={best_ge_n[1]}). Ward global clustering "
                           f"may be heavy -- consider a smaller --n-clusters if it OOMs.")
        else:
            logger.info(f"[auto-threshold] chosen threshold_init={best_ge_n[0]:.4g} (m={best_ge_n[1]})")
        return best_ge_n[0]
    logger.warning(f"[auto-threshold] could not reach {n} subclusters; using {thr:.4g}")
    return thr


def cluster_once(features, n_clusters, weighting_PCs=True, threshold_init=None):
    """Steps 3-4 (StandardScaler -> PCA -> BIRCH) exactly as in DIRECTSampler,
    so SelectKFromClusters(k) on the result equals a full DIRECTSampler run at
    that k. threshold_init=None calibrates the radius with _auto_threshold."""
    Xs = StandardScaler().fit_transform(features)
    pca = PrincipalComponentAnalysis(weighting_PCs=weighting_PCs).fit(Xs)
    pca_features = pca.transform(Xs)
    if threshold_init is None:
        print(f"[cluster] auto-calibrating BIRCH threshold for ~{n_clusters} clusters "
              f"on {pca_features.shape[0]} x {pca_features.shape[1]} features ...", file=sys.stderr)
        threshold_init = _auto_threshold(pca_features, n_clusters)
        print(f"[cluster] auto threshold_init = {threshold_init:.4g}", file=sys.stderr)
    print(f"[cluster] PCA kept {pca_features.shape[1]} PCs; running BIRCH on "
          f"{pca_features.shape[0]} points -> ~{n_clusters} clusters "
          f"(single-threaded; this is the slow step)...", file=sys.stderr)
    clustering = BirchClustering(n=n_clusters, threshold_init=threshold_init).transform(pca_features)
    return clustering


def _cluster_sizes(clustering):
    _, counts = np.unique(np.asarray(clustering["labels"]), return_counts=True)
    return np.sort(counts)


def _yield_fn(sizes_sorted):
    """k -> sum_c min(k, size_c), via a prefix sum over the sorted sizes."""
    prefix = np.concatenate(([0], np.cumsum(sizes_sorted, dtype=np.int64)))

    def yld(k):
        idx = int(np.searchsorted(sizes_sorted, k, side="left"))
        return int(prefix[idx] + k * (len(sizes_sorted) - idx))

    return yld


def pick_k_for_target(clustering, target):
    """The k whose yield is closest to target. Returns (k, size, sizes, yld)."""
    sizes = _cluster_sizes(clustering)
    yld = _yield_fn(sizes)
    max_k = int(sizes[-1])
    lo, hi = 1, max_k
    while lo < hi:
        mid = (lo + hi) // 2
        if yld(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    k_hi, k_lo = lo, max(1, lo - 1)
    y_hi, y_lo = yld(k_hi), yld(k_lo)
    if abs(y_lo - target) <= abs(y_hi - target):
        return k_lo, y_lo, sizes, yld
    return k_hi, y_hi, sizes, yld


def _report_clustering(sizes, yld, target, best_k, best_size):
    n_clusters = len(sizes)
    total = int(sizes.sum())
    pct = lambda q: int(np.percentile(sizes, q))
    print("\n[report] clustering summary", file=sys.stderr)
    print(f"  clusters formed   : {n_clusters}", file=sys.stderr)
    print(f"  points clustered  : {total}", file=sys.stderr)
    print(f"  cluster size      : min={int(sizes[0])}  p25={pct(25)}  "
          f"median={pct(50)}  mean={sizes.mean():.1f}  p75={pct(75)}  "
          f"max={int(sizes[-1])}", file=sys.stderr)
    ks = sorted(set([1, 5, 10, 20, 50, 100]
                    + [best_k - 2, best_k - 1, best_k, best_k + 1, best_k + 2]))
    ks = [k for k in ks if 1 <= k <= int(sizes[-1])]
    print(f"\n[report] yield(k) = subset size for each k  (target = {target}):", file=sys.stderr)
    for k in ks:
        y = yld(k)
        mark = "  <-- chosen" if k == best_k else ""
        print(f"    k={k:<5d} -> {y:>12d}  ({100.0 * y / target:6.1f}% of target){mark}",
              file=sys.stderr)
    dev = 100.0 * (best_size - target) / target
    print(f"\n[report] chosen k={best_k}: {best_size} structures "
          f"({dev:+.2f}% vs target {target})\n", file=sys.stderr)


def _write_outputs(selected, features, args):
    np.save(args.out + ".indices.npy", np.asarray(selected, dtype=np.int64))
    with open(args.out + ".indices.json", "w") as fh:
        json.dump(selected, fh)
    if args.ids:
        with open(args.ids) as fh:
            ids = json.load(fh)
        selected_ids = [ids[i] for i in selected]
        with open(args.out + ".ids.json", "w") as fh:
            json.dump(selected_ids, fh)
        print(f"[direct] wrote selected ids -> {args.out}.ids.json", file=sys.stderr)
    print(f"[done] indices -> {args.out}.indices.npy / .json", file=sys.stderr)


# --------------------------------------------------------------------------- #
#  Self-test and maml cross-check
# --------------------------------------------------------------------------- #
def _reference_center_selection(pca_features, labels, label_centers, k):
    """Plain implementation of step 5 ('center'), used to cross-check the core."""
    selected = []
    for label in set(labels):
        idx = np.where(labels == label)[0]
        d = np.linalg.norm(pca_features[idx] - label_centers[label], axis=1)
        order = idx[np.argsort(d, kind="stable")]
        ranks = [int(i) for i in np.linspace(0, len(order) - 1, k)]
        selected.extend(order[r] for r in ranks)
    return set(int(i) for i in selected)


def self_test():
    rng = np.random.RandomState(0)
    n_blobs, per_blob, dim = 6, 200, 8
    centers = rng.uniform(-20, 20, size=(n_blobs, dim))
    X = np.vstack([c + rng.randn(per_blob, dim) for c in centers])

    failures = []

    # k = 1: one pick per cluster, the one nearest the centroid.
    Xs = StandardScaler().fit_transform(X)
    pca = PrincipalComponentAnalysis(weighting_PCs=True).fit(Xs)
    feats = pca.transform(Xs)
    clus = BirchClustering(n=n_blobs).transform(feats)
    sel1 = SelectKFromClusters(k=1).transform(clus)["selected_indexes"]
    if len(set(sel1)) != n_blobs:
        failures.append(f"k=1 expected {n_blobs} unique picks, got {len(set(sel1))}")
    labels, lc = clus["labels"], clus["label_centers"]
    for i in sel1:
        lab = labels[i]
        idx = np.where(labels == lab)[0]
        d = np.linalg.norm(feats[idx] - lc[lab], axis=1)
        if idx[np.argmin(d)] != i:
            failures.append(f"k=1 pick {i} is not nearest-to-centroid in cluster {lab}")

    # k = 5 against the reference; argpartition and argsort may break exact
    # distance ties differently, so only non-tie differences fail.
    k = 5
    sel5 = set(int(i) for i in SelectKFromClusters(k=k).transform(clus)["selected_indexes"])
    ref5 = _reference_center_selection(feats, labels, lc, k)
    if sel5 != ref5:
        only = sel5.symmetric_difference(ref5)
        bad = []
        for i in only:
            lab = labels[i]
            idx = np.where(labels == lab)[0]
            d = np.linalg.norm(feats[idx] - lc[lab], axis=1)
            if np.sum(np.isclose(d, np.linalg.norm(feats[i] - lc[lab]))) < 2:
                bad.append(i)
        if bad:
            failures.append(f"k=5 selection differs from reference at non-tie indices {bad}")

    # Kaiser's rule.
    m_kept = feats.shape[1]
    m_expected = int(np.sum(pca.pca.explained_variance_ > 1))
    if m_kept != m_expected:
        failures.append(f"Kaiser rule: kept {m_kept} PCs, expected {m_expected}")

    # Determinism.
    sel_a, _ = run(X, n_clusters=n_blobs, k=3, seed=1)
    sel_b, _ = run(X, n_clusters=n_blobs, k=3, seed=1)
    if sel_a != sel_b:
        failures.append("pipeline is not deterministic for center selection")

    # n_clusters <= selected <= n_clusters * k.
    sel_c, _ = run(X, n_clusters=n_blobs, k=4)
    if not (n_blobs <= len(sel_c) <= n_blobs * 4):
        failures.append(f"selected count {len(sel_c)} outside [{n_blobs}, {n_blobs*4}]")

    for w in (True, False):
        s, _ = run(X, n_clusters=n_blobs, k=2, weighting_PCs=w)
        if len(s) < n_blobs:
            failures.append(f"weighting={w}: fewer selections ({len(s)}) than clusters")

    if failures:
        print("SELF-TEST FAILED:", file=sys.stderr)
        for f in failures:
            print("  -", f, file=sys.stderr)
        sys.exit(1)
    print("SELF-TEST PASSED: DIRECT core (PCA/Kaiser + BIRCH + stratified-center) "
          "reproduces the paper's step 3-5 protocol and matches the reference.")


def verify_vs_maml(n_clusters=40, k=7, seed=0):
    """Run this DIRECTSampler and maml's DIRECTSampler(structure_encoder=False)
    on the same features and require identical selections (needs maml)."""
    try:
        from maml.sampling.clustering import BirchClustering as MamlBirch
        from maml.sampling.direct import DIRECTSampler as MamlDIRECT
        from maml.sampling.stratified_sampling import SelectKFromClusters as MamlSelectK
    except Exception as exc:  # noqa: BLE001
        print(f"maml not importable ({exc}); `pip install maml` to run this check.", file=sys.stderr)
        sys.exit(2)

    rng = np.random.RandomState(seed)
    dim = 64
    centers = rng.uniform(-15, 15, size=(50, dim))
    X = np.vstack([c + rng.randn(rng.randint(30, 120), dim) for c in centers]).astype(np.float64)

    mine, _ = run(X, n_clusters=n_clusters, k=k)

    maml_sampler = MamlDIRECT(
        structure_encoder=False,
        clustering=MamlBirch(n=n_clusters),
        select_k_from_clusters=MamlSelectK(k=k),
    )
    maml_out = maml_sampler.fit_transform(X)
    theirs = sorted(int(i) for i in maml_out["selected_indexes"])

    if mine == theirs:
        print(f"VERIFY PASSED: identical selection ({len(mine)} structures) "
              f"from ours and maml.sampling.DIRECTSampler (n={n_clusters}, k={k}).")
    else:
        only_mine = sorted(set(mine) - set(theirs))
        only_theirs = sorted(set(theirs) - set(mine))
        print("VERIFY FAILED: selections differ.", file=sys.stderr)
        print(f"  ours={len(mine)} theirs={len(theirs)} "
              f"only_ours={only_mine[:20]} only_theirs={only_theirs[:20]}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="DIRECT steps 3-5: select a subset from cached features.")
    ap.add_argument("--features", help="(N, D) .npy feature matrix (stack_shards.py -> feats.npy)")
    ap.add_argument("--ids", default=None, help="optional .ids.json mapping rows -> identifiers")
    ap.add_argument("--n-clusters", type=int, default=None, help="BIRCH cluster count n")
    ap.add_argument("--k", type=int, default=1, help="structures per cluster (fixed-k mode)")
    ap.add_argument("--target-size", type=int, default=None,
                    help="desired subset size: cluster once at --n-clusters and choose the k whose "
                         "subset size is closest (--k is ignored). Requires --n-clusters.")
    ap.add_argument("--threshold-init", type=float, default=None,
                    help="initial BIRCH radius threshold; default: auto-calibrated in --target-size "
                         "mode, maml's 0.5 in fixed-k mode.")
    ap.add_argument("--no-weighting", action="store_true", help="disable explained-variance PC weighting")
    ap.add_argument("--allow-duplicate", action="store_true", help="keep duplicate selections (k > cluster size)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="direct_selected", help="output prefix")
    ap.add_argument("--self-test", action="store_true", help="run the built-in self-test and exit")
    ap.add_argument("--verify-vs-maml", action="store_true",
                    help="require identical selection to maml.sampling.DIRECTSampler (needs maml)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.verify_vs_maml:
        verify_vs_maml()
        return

    if not args.features:
        ap.error("--features is required (or use --self-test)")

    features = np.load(args.features)
    print(f"[load] features {features.shape} from {args.features}", file=sys.stderr)

    if args.target_size is not None:
        if args.n_clusters is None:
            ap.error("--target-size requires --n-clusters (the cluster count n is fixed; k is solved).")
        clustering = cluster_once(
            features,
            n_clusters=args.n_clusters,
            weighting_PCs=not args.no_weighting,
            threshold_init=args.threshold_init,
        )
        best_k, best_size, sizes, yld = pick_k_for_target(clustering, args.target_size)
        _report_clustering(sizes, yld, args.target_size, best_k, best_size)
        out = SelectKFromClusters(k=best_k, allow_duplicate=False).transform(clustering)
        selected = sorted(int(i) for i in out["selected_indexes"])
        print(f"[direct] selected {len(selected)} / {features.shape[0]} structures "
              f"(n={args.n_clusters}, k={best_k})", file=sys.stderr)
    else:
        selected, _ = run(
            features,
            n_clusters=args.n_clusters,
            k=args.k,
            weighting_PCs=not args.no_weighting,
            threshold_init=(args.threshold_init if args.threshold_init is not None else 0.5),
            allow_duplicate=args.allow_duplicate,
            seed=args.seed,
        )
        print(f"[direct] selected {len(selected)} / {features.shape[0]} structures", file=sys.stderr)

    _write_outputs(selected, features, args)


if __name__ == "__main__":
    main()
