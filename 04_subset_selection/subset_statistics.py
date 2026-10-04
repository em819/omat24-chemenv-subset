#!/usr/bin/env python3
"""
Size and DFT force statistics of training subsets (extxyz).

Backs the subset comparison at a fixed budget of ~6.5 M atoms:
  * main text, Methods: EnvTok ~796 k structures with ~8 atoms/structure;
    DIRECT and random 356-387 k structures with 17-18 atoms/structure;
    DIRECT shifts the force distribution to higher forces, EnvTok has the
    broadest force distribution;
  * SI Sec. S2.5 and Fig. S1: random median ~1.1 eV/A; DIRECT median shift and
    high-force tail shift; EnvTok widest 5th-95th percentile range; the
    percentile ranges in which DIRECT's force quantiles lie above EnvTok's
    (--compare DIRECT EnvTok).
The comparison with the full OMat24 force distribution ("random reproduces the
parent distribution") is not computed here; pass the full training split as one
more --subset to obtain it.

Each --subset gives a label and one or more extxyz files. All files under one
label are POOLED (e.g. the four random seeds): its percentiles are those of the
union of all atoms, not an average over files. Every file is also reported on
its own, so the seed-to-seed spread is visible.

The statistic is the per-atom force norm |F_i| = sqrt(Fx^2 + Fy^2 + Fz^2) in
eV/A, read from the 'forces' column. Its position in the atom line is taken
from each frame's Properties= key, because OMat24 frames come in two layouts
(species:S:1:pos:R:3:forces:R:3 and
 species:S:1:pos:R:3:initial_magmoms:R:1:forces:R:3).
Nothing is subsampled: every atom of every file enters the statistics.
Percentiles use numpy's default ('linear') definition, on all atoms including
any with |F| = 0 (--exclude-zero drops those from the force statistics, as the
log-scale figure does; the structure and atom counts always include them).
Per-structure statistics of the largest atomic force in each structure,
max_i |F_i|, are reported as well (columns smax_*), for statements about
high-force structures rather than atoms.

Works directly on the training files the pipeline writes:
  stage 05                 -> subset_train.xyz           (EnvTok)
  04_subset_selection/DIRECT  -> subset_DIRECT_train.xyz
  04_subset_selection/random  -> random_seed<seed>/train.xyz

USAGE
  python subset_statistics.py \
      --subset EnvTok  subset_train.xyz \
      --subset DIRECT  subset_DIRECT_train.xyz \
      --subset random  random_seed1/train.xyz random_seed2/train.xyz \
                       random_seed3/train.xyz random_seed4/train.xyz \
      --reference random --compare DIRECT EnvTok \
      --output-csv subset_statistics.csv

Memory: |F_i| is held as float64 for every atom of every file (8 bytes/atom),
plus a pooled copy per multi-file label; the four random seeds and the two other
subsets of the paper (~52 M atoms) need ~0.8 GB.
"""

import argparse
import csv
import os
import re
import time
from multiprocessing import Pool

import numpy as np

# =============================================================================
#                            USER CONFIGURATION
# =============================================================================
# Percentiles reported for |F_i| (numpy 'linear' definition).
PERCENTILES = [1, 5, 25, 50, 75, 95, 99, 99.9]
# Fraction of atoms with |F_i| strictly above each of these values (eV/A).
THRESHOLDS = [5.0, 10.0, 20.0]
# Name of the per-atom force property in the Properties= key.
FORCES_KEY = "forces"
# Step (in percent) of the percentile grid used to locate quantile crossings
# for --compare A B (where A's force quantile overtakes B's).
CROSSING_GRID_STEP = 0.1
# Worker processes; each parses whole files.
NPROC = 8
# =============================================================================

PROPERTIES_RE = re.compile(r'Properties=("?)([^\s"]+)\1')


def forces_layout(comment, forces_key):
    """(column of Fx, total number of columns) from a frame's Properties= key."""
    m = PROPERTIES_RE.search(comment)
    if m is None:
        raise ValueError(f"no Properties= key in comment line: {comment[:200]!r}")
    fields = m.group(2).split(":")
    if len(fields) % 3:
        raise ValueError(f"malformed Properties= key: {m.group(2)!r}")
    col, fcol = 0, None
    for name, typ, cnt in zip(fields[0::3], fields[1::3], fields[2::3]):
        cnt = int(cnt)
        if name == forces_key:
            if typ != "R" or cnt != 3:
                raise ValueError(f"'{forces_key}' is {typ}:{cnt}, expected R:3")
            fcol = col
        col += cnt
    if fcol is None:
        raise ValueError(f"no '{forces_key}' column in Properties={m.group(2)}")
    return fcol, col


def parse_file(task):
    """Stream one extxyz file -> per-atom |F| (float64) and counts."""
    path, forces_key = task
    t0 = time.time()
    chunks = []
    frame_max = []                   # largest |F_i| of each structure
    n_frames = 0
    layouts = {}                     # Properties key -> (fcol, ncol)
    layout_counts = {}               # Properties key -> number of frames
    with open(path, "r") as f:
        lineno = 0
        while True:
            header = f.readline()
            lineno += 1
            if not header:
                break
            if not header.strip():   # tolerate blank lines between frames
                continue
            try:
                n = int(header.split()[0])
            except ValueError:
                raise ValueError(f"{path}:{lineno}: expected an atom count, "
                                 f"got {header[:80]!r}")
            comment = f.readline()
            lines = [f.readline() for _ in range(n)]
            lineno += n + 1
            m = PROPERTIES_RE.search(comment)
            key = m.group(2) if m else None
            if key not in layouts:
                layouts[key] = forces_layout(comment, forces_key)
            fcol, ncol = layouts[key]
            layout_counts[key] = layout_counts.get(key, 0) + 1
            toks = " ".join(lines).split()
            if len(toks) != n * ncol:
                raise ValueError(f"{path}: frame {n_frames + 1} (ending line "
                                 f"{lineno}) has {len(toks)} tokens, expected "
                                 f"{n} atoms x {ncol} columns")
            # tokens are row-major (atom, column): every ncol-th token from fcol
            fx = np.array(toks[fcol::ncol], dtype=np.float64)
            fy = np.array(toks[fcol + 1::ncol], dtype=np.float64)
            fz = np.array(toks[fcol + 2::ncol], dtype=np.float64)
            fn = np.sqrt(fx * fx + fy * fy + fz * fz)
            chunks.append(fn)
            frame_max.append(fn.max() if n else 0.0)
            n_frames += 1
    if n_frames == 0:
        raise ValueError(f"{path}: no frames read")
    fnorm = np.concatenate(chunks)
    return dict(path=path, n_frames=n_frames, fnorm=fnorm,
                fmax=np.asarray(frame_max, dtype=np.float64),
                layouts=layout_counts, seconds=time.time() - t0)


def describe(fnorm_all, fmax, percentiles, thresholds, exclude_zero):
    """One row of the output table.

    fnorm_all: per-atom |F_i| of all atoms (counts are taken from it);
    fmax: largest |F_i| per structure. With exclude_zero the per-atom force
    statistics use only atoms with |F_i| > 0.
    """
    n_frames = fmax.size
    n_atoms = fnorm_all.size
    fnorm = fnorm_all[fnorm_all > 0] if exclude_zero else fnorm_all
    pct = np.percentile(fnorm, percentiles, method="linear")
    p = dict(zip(percentiles, pct))
    row = {
        "n_structures": n_frames,
        "n_atoms": n_atoms,
        "atoms_per_structure": n_atoms / n_frames if n_frames else float("nan"),
        "n_zero_force_atoms": int((fnorm_all == 0.0).sum()),
        "mean": float(fnorm.mean()),
    }
    for q in percentiles:
        row[f"p{q:g}"] = float(p[q])
    row["max"] = float(fnorm.max())
    if 5 in p and 95 in p:
        # width of the 5th-95th percentile range, linear and on the log axis
        row["p95_minus_p5"] = float(p[95] - p[5])
        row["log10_p95_over_p5"] = (float(np.log10(p[95] / p[5]))
                                    if p[5] > 0 else float("nan"))
    for t in thresholds:
        row[f"frac_gt_{t:g}"] = float((fnorm > t).mean())
    # per structure: largest atomic force
    for q, v in zip(percentiles, np.percentile(fmax, percentiles, method="linear")):
        row[f"smax_p{q:g}"] = float(v)
    for t in thresholds:
        row[f"smax_frac_gt_{t:g}"] = float((fmax > t).mean())
    return row


def quantile_crossings(a, b, step):
    """Percentile levels where the quantile curves of a and b cross.

    Returns (grid, diff, list of (q_lo, q_hi) intervals where diff > 0 switches
    between true and false), with diff = Q_a(q) - Q_b(q).
    """
    grid = np.round(np.arange(step, 100.0, step), 6)
    diff = (np.percentile(a, grid, method="linear")
            - np.percentile(b, grid, method="linear"))
    s = diff > 0                     # A above B; an exact tie counts as "not above"
    flips = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
             if s[i] != s[i + 1]]
    return grid, diff, flips


def fmt(v):
    if v == "":
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:.4g}" if abs(v) < 1e-2 and v != 0 else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", nargs="+", action="append", required=True,
                    metavar="LABEL XYZ",
                    help="a label followed by one or more extxyz files; files "
                         "under one label are pooled. Repeat per subset.")
    ap.add_argument("--percentiles", type=float, nargs="+", default=PERCENTILES)
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS,
                    help="report the fraction of atoms with |F| above each (eV/A).")
    ap.add_argument("--reference", default=None,
                    help="label to subtract from every other label "
                         "(differences of all statistics).")
    ap.add_argument("--compare", nargs=2, action="append", default=[],
                    metavar=("A", "B"),
                    help="locate the percentiles where Q_A(q) - Q_B(q) changes "
                         "sign. Repeatable.")
    ap.add_argument("--crossing-step", type=float, default=CROSSING_GRID_STEP)
    ap.add_argument("--forces-key", default=FORCES_KEY)
    ap.add_argument("--exclude-zero", action="store_true",
                    help="drop atoms with |F| == 0 before computing statistics "
                         "(what a log-scale plot implicitly does).")
    ap.add_argument("--nproc", type=int, default=NPROC)
    ap.add_argument("--output-csv", default="subset_statistics.csv")
    args = ap.parse_args()

    percentiles = [float(q) for q in args.percentiles]
    percentiles = [int(q) if q.is_integer() else q for q in percentiles]

    subsets = {}
    for entry in args.subset:
        if len(entry) < 2:
            ap.error(f"--subset {entry[0]}: give at least one file")
        label, files = entry[0], entry[1:]
        if label in subsets:
            ap.error(f"label '{label}' given twice")
        for p in files:
            if not os.path.isfile(p):
                ap.error(f"no such file: {p}")
        subsets[label] = files
    for lab in ([args.reference] if args.reference else []) + \
            [x for pair in args.compare for x in pair]:
        if lab not in subsets:
            ap.error(f"unknown label '{lab}'")

    # ---- parse every distinct file once, in parallel -------------------------
    unique = list(dict.fromkeys(p for files in subsets.values() for p in files))
    parsed = {}
    with Pool(max(1, min(args.nproc, len(unique)))) as pool:
        for r in pool.imap_unordered(parse_file,
                                     [(p, args.forces_key) for p in unique]):
            parsed[r["path"]] = r
            lay = ", ".join(f"{k}: {v:,}" for k, v in r["layouts"].items())
            print(f"parsed {r['path']}: {r['n_frames']:,} frames, "
                  f"{r['fnorm'].size:,} atoms [{r['seconds']:.0f}s]\n"
                  f"    layouts: {lay}", flush=True)

    def pooled_arrays(files):
        if len(files) == 1:
            return parsed[files[0]]["fnorm"], parsed[files[0]]["fmax"]
        return (np.concatenate([parsed[p]["fnorm"] for p in files]),
                np.concatenate([parsed[p]["fmax"] for p in files]))

    # ---- statistics per file and per (pooled) label --------------------------
    rows = []
    pooled = {}
    for label, files in subsets.items():
        for p in files:
            r = describe(parsed[p]["fnorm"], parsed[p]["fmax"], percentiles,
                         args.thresholds, args.exclude_zero)
            rows.append(dict(kind="file", label=label, file=p, **r))
        allf, allmax = pooled_arrays(files)
        if any(label in pair for pair in args.compare):
            # kept only where a crossing is requested
            pooled[label] = allf[allf > 0] if args.exclude_zero else allf
        r = describe(allf, allmax, percentiles, args.thresholds, args.exclude_zero)
        rows.append(dict(kind="label", label=label,
                         file=";".join(files) if len(files) > 1 else files[0], **r))

    if args.reference:
        ref = next(r for r in rows if r["kind"] == "label"
                   and r["label"] == args.reference)
        # differences of the intensive statistics only (pooled counts of a
        # multi-file label are not comparable with a single file's)
        stat_keys = [k for k in ref if k not in ("kind", "label", "file",
                                                 "n_structures", "n_atoms",
                                                 "n_zero_force_atoms")]
        for r in [r for r in rows if r["kind"] == "label"
                  and r["label"] != args.reference]:
            rows.append(dict(kind="difference",
                             label=f"{r['label']} - {args.reference}", file="",
                             **{k: r[k] - ref[k] for k in stat_keys}))
    blank = dict.fromkeys(["n_structures", "n_atoms", "n_zero_force_atoms"], "")
    rows = [{**blank, **r} if r["kind"] == "difference" else r for r in rows]

    # ---- table ---------------------------------------------------------------
    cols = ["n_structures", "n_atoms", "atoms_per_structure", "mean"] + \
        [f"p{q:g}" for q in percentiles] + \
        [c for c in ("p95_minus_p5", "log10_p95_over_p5") if c in rows[0]] + \
        [f"frac_gt_{t:g}" for t in args.thresholds] + \
        [f"smax_p{q:g}" for q in percentiles if q in (50, 95, 99)] + \
        [f"smax_frac_gt_{t:g}" for t in args.thresholds]
    print(f"\nPer-atom |F| statistics (eV/A), "
          f"{'atoms with |F| = 0 excluded' if args.exclude_zero else 'all atoms'}, "
          f"numpy 'linear' percentiles")
    head = f"{'kind':10s} {'label':24s} {'file':28s} " + \
        " ".join(f"{c[:12]:>12s}" for c in cols)
    print(head)
    print("-" * len(head))
    for r in rows:
        name = os.path.basename(r["file"]) if r["kind"] == "file" else \
            (f"{len(r['file'].split(';'))} files pooled" if ";" in r["file"] else "")
        print(f"{r['kind']:10s} {r['label'][:24]:24s} {name[:28]:28s} " +
              " ".join(f"{fmt(r[c]):>12s}" for c in cols))

    # ---- quantile crossings --------------------------------------------------
    cross_rows = []
    for a, b in args.compare:
        grid, diff, flips = quantile_crossings(pooled[a], pooled[b],
                                               args.crossing_step)
        above = grid[diff > 0]
        print(f"\nQ_{a}(q) - Q_{b}(q) on a {args.crossing_step:g}% grid: "
              f"positive at {above.size} of {grid.size} levels")
        if not flips:
            print("    no sign change on the grid")
        for lo, hi in flips:
            i = int(np.searchsorted(grid, lo))
            print(f"    sign change between p{lo:g} ({diff[i]:+.4f}) and "
                  f"p{hi:g} ({diff[i + 1]:+.4f})")
            cross_rows.append(dict(A=a, B=b, q_lo=lo, q_hi=hi,
                                   diff_at_q_lo=diff[i], diff_at_q_hi=diff[i + 1]))
        if diff[-1] > 0:
            # last level from which A stays above B up to the top of the grid
            neg = np.nonzero(diff <= 0)[0]
            start = grid[neg[-1] + 1] if neg.size else grid[0]
            print(f"    {a} > {b} for every grid level from p{start:g} upwards")

    # ---- CSV -----------------------------------------------------------------
    out = args.output_csv
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fields = ["kind", "label", "file", "n_structures", "n_atoms",
              "atoms_per_structure", "n_zero_force_atoms"] + \
        [k for k in rows[0] if k not in ("kind", "label", "file", "n_structures",
                                         "n_atoms", "atoms_per_structure",
                                         "n_zero_force_atoms")]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    if cross_rows:
        cout = os.path.splitext(out)[0] + "_quantile_crossings.csv"
        with open(cout, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cross_rows[0]))
            w.writeheader()
            w.writerows(cross_rows)
        print(f"wrote {cout}")


if __name__ == "__main__":
    main()
