#!/usr/bin/env python3
"""
Stage 09 / MatPES -- where the extreme per-atom force errors come from, and
the bulk half of Table I with the pure-nitrogen cells taken out.

Environment: envB-mace (imports eval_with_report_efs.py from this directory,
which imports mace). CPU only.

It reads the same per-model `*_paper` directories as `eval_with_report_efs.py
--aggregate` (per_structure.npz + per_atom_forces.npz of each model, all
evaluated on the same MatPES file) plus that MatPES extxyz file itself, and
computes two things.

PART A -- BLOW-UP ATTRIBUTION. For every model, the atoms with
||dF_i|| > --threshold (default 10 eV/A, strict, non-finite counted as +inf)
in the bulk + non-bulk population (overcompressed cells excluded, as in
paper_tables/blowup_atoms_bulk_plus_non_bulk.csv), and what share of them
  * sits in a structure containing at least one bonded fragment of periodic
    dimensionality 0 or 1 (isolated cluster/molecule or chain), and
  * sits in a cell made only of --pure-element (default N).
Backs, in the main text after Table I:
  "Amongst the over 3 million atoms in MatPES, extreme force errors
   (>10 eV/A) are observed in 4150 atoms for MACE-OMAT-0, compared with 1671
   for DIRECT, 4428 for EnvTok, and 2924-6226 for random selection. For the
   subset-trained models, 90-96% of these atoms lie in structures containing
   isolated 0D or 1D fragments. For MACE-OMAT-0, 97% occur in highly
   compressed polymeric nitrogen phases."
"Polymeric nitrogen" is operationalised as "every atom of the cell is N"
(no density or bonding test); the printout also gives, for those cells, how
many are bulk (3-periodic bonded network) and their volume per atom, so the
"highly compressed polymeric" description can be checked rather than assumed.

PART B -- BULK TABLE I WITHOUT PURE-N CELLS. The bulk per-atom binned profile
of Table I, once for all bulk cells and once with the pure-element cells
removed, with Table I's conventions:
  * bins cut on the CEILING model's (MACE-OMAT-0's) own per-atom ||dF_i||,
    half-open [lo, hi), edges PERATOM_CEILING_BINS of the eval script;
  * entry = median ||dF_i|| over the bin's atoms (_wquantile, the lower
    median, as in the aggregate); bins under PERATOM_MIN_BIN_ATOMS skipped;
  * random = mean +- sample s.d. (ddof=1) over the rdm* replicas;
  * best = lowest of the candidate columns (every non-seed, non-ceiling model,
    plus the random mean), only if its margin over the next-lowest exceeds
    the seed s.d. (the bold rule of Table I); '-' otherwise.
Backs:
  "EnvTok trails DIRECT and random selection in the two highest bins only
   because of fewer than 100 structures of highly compressed polymeric
   nitrogen, which contribute 11% and 82% of their atoms, respectively.
   Without them, EnvTok is the most accurate subset-trained model in every
   bulk bin."

Domains are those of the paper tables: the overcompressed screen stored at
evaluation time is kept, and bulk / non-bulk are re-derived from the stored
descriptors with --non-bulk-detector (default 'dim', as in
submit_aggregate.sh), using the eval script's own _rederive_domains.
Fragment dimensionalities are computed with the eval script's own bond graph
(_bonded_pairs at NON_BULK_BOND_SCALE) and dimensionality code
(structure_descriptors, applied to each bonded fragment on its own), and are
checked against the 'dim' and 'n_fragments' columns stored at evaluation.

    python blowup_attribution.py \\
        --models v14_paper omat0_paper direct_paper rdm1_paper rdm2_paper \\
                 rdm3_paper rdm4_paper \\
        --labels v14 OMAT0 DIRECT rdm1 rdm2 rdm3 rdm4 --ceiling OMAT0 \\
        --configs matpes_pbe_2025.2_no_rare_earth.xyz --out-dir blowup_out
"""

import argparse
import csv
import io
import os
import sys
from collections import Counter

import numpy as np

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --models     REQUIRED. One `*_paper` directory per model, as written by
#                eval_with_report_efs.py (submit_eval_single.sh): each must
#                hold per_structure.npz AND per_atom_forces.npz. The same list
#                submit_aggregate.sh passes to --aggregate. The paper used
#                seven: EnvTok (v14), MACE-OMAT-0, DIRECT, rdm1..rdm4.
#
#   --labels     One label per --models entry, same order. A label starting
#                with rdm / seed / random marks a random-seed replica (the
#                eval script's SEED_LABEL_PREFIXES); their spread is the noise
#                floor of the bold rule. Default: the directory names.
#
#   --ceiling    REQUIRED. The LABEL (one of --labels) of the full-data
#                reference model the Table I bins are cut on (MACE-OMAT-0).
#
#   --configs    REQUIRED. The MatPES extxyz FILE the models were evaluated on
#                (the --configs of eval_with_report_efs.py; the paper used
#                MatPES PBE 2025.2 without rare earths, 323,096 frames). Read
#                only for the frames that hold a blown-up atom, located by the
#                per_structure.npz 'index' column and checked against 'sid'.
#
#   --threshold  Per-atom force-error norm threshold, eV/A, strict (>).
#                Default 10.
#
#   --pure-element  Element symbol whose single-element cells are singled out
#                ("polymeric nitrogen" in the paper). Default N.
#
#   --non-bulk-detector  Bulk / non-bulk criterion re-derived from the stored
#                descriptors, as --aggregate_non_bulk_detector. Default 'dim'
#                (the paper's Table I). 'none' keeps the stored split.
#
#   --out-dir    Where the two CSVs are written (created if missing):
#                blowup_attribution.csv, bulk_profile_pure_element_excluded.csv
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_with_report_efs as E  # noqa: E402  (repo producer; reused, not copied)


# ---------------------------------------------------------------------------
# Loading: the aggregate's own loaders, alignment and domain re-derivation
# ---------------------------------------------------------------------------
def load_models(paths, labels, detector):
    """Per-structure tables aligned to the first model, domains re-derived,
    and every model's per-atom err_norm permuted into the common atom order."""
    tabs = [E.load_per_structure(p) for p in paths]
    orders, aligned = [], []
    for lab, t in zip(labels, tabs):
        _, it = E.align_to(tabs[0], t)
        orders.append(it)
        aligned.append({k: v[it] for k, v in t.items()
                        if isinstance(v, np.ndarray)
                        and v.shape[:1] == t['sid'].shape})
    if detector != 'none':
        E._rederive_domains(list(zip(labels, aligned)), detector)
    base = aligned[0]
    # Atoms per structure in the common order, from the first model's per-atom
    # file as in the aggregate (0 for a structure that carried no forces).
    with np.load(E._per_atom_forces_path(paths[0])) as z:
        base_counts = np.diff(z['struct_offset'].astype(np.int64))[orders[0]]

    err, z0 = {}, None
    for lab, path, tab, it in zip(labels, paths, tabs, orders):
        pa = E.load_per_atom_forces(path, rows=tab)
        if pa is None:
            raise SystemExit('no per_atom_forces.npz for %r (%s)' % (lab, path))
        perm = E._peratom_base_perm(pa, it, base_counts, path)
        z = np.asarray(pa['Z'])[perm]
        if z0 is None:
            z0 = z
        elif not np.array_equal(z, z0):
            raise SystemExit('per-atom Z of %r differs from %r after matching: '
                             'not evaluated on the same file' % (lab, labels[0]))
        # Non-finite -> +inf, the aggregate's convention: a divergent atom
        # counts above every threshold and ranks worst in every median.
        err[lab] = E._worst_fill(pa['err_norm'].astype(np.float64))[perm]
        del pa
    row = np.repeat(np.arange(len(base_counts)), base_counts)
    base['atom_offset'] = np.concatenate(([0], np.cumsum(base_counts)))
    return base, err, row, z0


# ---------------------------------------------------------------------------
# Fragment dimensionalities, from the eval script's bond graph
# ---------------------------------------------------------------------------
def fragment_dims(atoms):
    """(fragment label per atom, periodic dimensionality per fragment).

    Fragments = connected components of the eval script's bond graph. The
    dimensionality of each is structure_descriptors()['dim'] of that fragment
    alone: a fragment's bonds are exactly the bonds of its atoms, so cutting
    it out changes nothing in its graph, and the eval script's per-cell 'dim'
    is the maximum over these.
    """
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    from ase.data import covalent_radii

    n = len(atoms)
    i, j, _S, _d = E._bonded_pairs(atoms, E.NON_BULK_BOND_SCALE,
                                   covalent_radii[atoms.numbers])
    adj = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    n_comp, lab = csg.connected_components(adj, directed=False)
    dims = np.array([E.structure_descriptors(atoms[lab == c])['dim']
                     for c in range(n_comp)], dtype=int)
    return lab, dims


def read_frames(configs, frame_idx):
    """{frame index: Atoms} for the requested frames only (byte offsets)."""
    import ase.io
    offs = E.build_frame_offsets(configs)
    out = {}
    with open(configs, 'rb') as f:
        for k in sorted(set(int(x) for x in frame_idx)):
            f.seek(offs[k])
            end = offs[k + 1] if k + 1 < len(offs) else None
            raw = f.read(end - offs[k]) if end is not None else f.read()
            # ASE parses the Properties= header, so both OMat24/MatPES column
            # layouts (with or without initial_magmoms) are read correctly.
            out[k] = ase.io.read(io.StringIO(raw.decode()), format='extxyz')
    return out


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def part_a(base, err, row, z, labels, ceiling, configs, thr, pure, out_dir):
    over = base['domain'] == 'overcompressed'
    bulk_s = base['domain'] == 'bulk'
    pop_s = (base['domain'] == 'bulk') | (base['domain'] == 'non_bulk')
    pure_s = base['elements'] == pure
    pop = pop_s[row]

    blow = {L: np.flatnonzero(pop & (err[L] > thr)) for L in labels}
    rows_needed = np.unique(np.concatenate([row[b] for b in blow.values()]))
    frames = read_frames(configs, base['index'][rows_needed])

    geo, n_mismatch = {}, 0
    off = base['atom_offset']
    for r in rows_needed:
        at = frames[int(base['index'][r])]
        # The id the eval script stored for this frame (matpes_id for MatPES).
        sid = E._PerStructure._sid(at, int(base['index'][r]))
        if sid != base['sid'][r]:
            raise SystemExit('frame %d of --configs is %s, per_structure.npz '
                             'says %s' % (base['index'][r], sid, base['sid'][r]))
        if not np.array_equal(at.numbers, z[off[r]:off[r + 1]]):
            raise SystemExit('atoms of %s differ between --configs and '
                             'per_atom_forces.npz' % base['sid'][r])
        lab, dims = fragment_dims(at)
        if dims.max() != base['dim'][r] or len(dims) != base['n_fragments'][r]:
            n_mismatch += 1
        geo[r] = (lab, dims)

    print('=' * 78)
    print('PART A -- atoms with ||dF_i|| > %g eV/A, bulk + non-bulk '
          '(%d structures, %d atoms; %d overcompressed cells excluded)'
          % (thr, pop_s.sum(), pop.sum(), over.sum()))
    print('  %d structures hold such an atom in at least one model; fragment '
          'graph disagrees with stored dim/n_fragments in %d of them'
          % (len(rows_needed), n_mismatch))
    if n_mismatch:
        raise SystemExit('the fragment analysis disagrees with the dim/n_fragments '
                         'stored by eval_with_report_efs.py in %d structures: '
                         'were --configs and the *_paper directories produced '
                         'together?' % n_mismatch)
    print('  has0D1D = atom in a structure with a 0D or 1D fragment; '
          'pure%s = atom in an all-%s cell' % (pure, pure))
    hdr = ('label', 'n_atoms', 'n_structures', 'bulk', 'non_bulk',
           'has0D1D', 'pct_has0D1D', 'pure', 'pct_pure', 'pure_bulk',
           'pct_pure_bulk', 'own_fragment_dim_0_1_2_3')
    print('  %-8s %7s %6s %6s %8s %7s %6s %6s %6s %9s %6s  %s' % hdr)
    recs = []
    for L in labels:
        b = blow[L]
        r_of = row[b]
        has01 = np.array([(geo[r][1] <= 1).any() for r in r_of], dtype=bool)
        own = Counter(int(geo[r][1][geo[r][0][k - off[r]]])
                      for k, r in zip(b, r_of))
        n = len(b)
        pct = (lambda x: 100.0 * x / n if n else float('nan'))
        p, pb = pure_s[r_of], pure_s[r_of] & bulk_s[r_of]
        rec = dict(label=L, role=('ceiling' if L == ceiling else
                                  'random' if E._is_seed_label(L) else 'subset'),
                   n_atoms=n, n_structures=len(np.unique(r_of)),
                   bulk=int(bulk_s[r_of].sum()),
                   non_bulk=int((~bulk_s[r_of]).sum()),
                   has0D1D=int(has01.sum()), pct_has0D1D=pct(has01.sum()),
                   pure=int(p.sum()), pct_pure=pct(p.sum()),
                   pure_bulk=int(pb.sum()), pct_pure_bulk=pct(pb.sum()),
                   **{'own_fragment_dim%d' % d: own.get(d, 0) for d in range(4)})
        recs.append(rec)
        print('  %-8s %7d %6d %6d %8d %7d %6.1f %6d %6.1f %9d %6.1f  %s'
              % (L, n, rec['n_structures'], rec['bulk'], rec['non_bulk'],
                 rec['has0D1D'], rec['pct_has0D1D'], rec['pure'],
                 rec['pct_pure'], rec['pure_bulk'], rec['pct_pure_bulk'],
                 '/'.join(str(own.get(d, 0)) for d in range(4))))

    seeds = [r['n_atoms'] for r in recs if r['role'] == 'random']
    sub = [r['pct_has0D1D'] for r in recs if r['role'] != 'ceiling']
    c = next(r for r in recs if r['role'] == 'ceiling')
    print('\n  summary: ceiling %s %d atoms; %s; random %s'
          % (ceiling, c['n_atoms'],
             ', '.join('%s %d' % (r['label'], r['n_atoms']) for r in recs
                       if r['role'] == 'subset'),
             '%d-%d' % (min(seeds), max(seeds)) if seeds else 'n/a'))
    if sub:
        print('  subset-trained models, share in structures with a 0D/1D '
              'fragment: %.1f-%.1f %%' % (min(sub), max(sub)))
    print('  ceiling, share in all-%s cells: %.1f %% (bulk all-%s cells: '
          '%.1f %%)' % (pure, c['pct_pure'], pure, c['pct_pure_bulk']))

    # The cells behind the ceiling's pure-element blow-ups, described rather
    # than assumed: periodic dimensionality and volume per atom.
    pr = np.unique(row[blow[ceiling]][pure_s[row[blow[ceiling]]]])
    if len(pr):
        vpa = base['volume_per_atom'][pr]
        print('  ceiling\'s all-%s cells with a blow-up: %d (dim 3: %d); '
              'volume/atom min %.2f median %.2f max %.2f A^3'
              % (pure, len(pr), int((base['dim'][pr] == 3).sum()),
                 np.nanmin(vpa), np.nanmedian(vpa), np.nanmax(vpa)))

    if not recs:
        raise SystemExit('no model rows to write')
    path = os.path.join(out_dir, 'blowup_attribution.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print('  wrote %s' % path)


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def part_b(base, err, row, labels, ceiling, pure, out_dir):
    seeds = [L for L in labels if E._is_seed_label(L)]
    cands = [L for L in labels if L != ceiling and L not in seeds]
    bulk_s = base['domain'] == 'bulk'
    pure_s = base['elements'] == pure
    bulk, pure_a = bulk_s[row], pure_s[row]
    cerr = err[ceiling]
    edges = list(E.PERATOM_CEILING_BINS)
    bins = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])] + \
        [(edges[-1], np.inf)]

    print('\n' + '=' * 78)
    print('PART B -- bulk, median ||dF_i|| [meV/A] per bin of %s\'s own '
          '||dF_i||' % ceiling)
    print('  bulk: %d structures, %d atoms; all-%s bulk cells: %d'
          % (bulk_s.sum(), bulk.sum(), pure, (bulk_s & pure_s).sum()))
    top = bulk & pure_a & (cerr >= edges[-2])
    print('  distinct all-%s bulk cells with atoms in the two highest bins: %d'
          % (pure, len(np.unique(row[top]))))
    recs = []
    for excl in (False, True):
        print('\n  -- %s' % ('all-%s cells EXCLUDED' % pure if excl
                             else 'all bulk cells'))
        # %pure and cells always describe the full bin (all bulk cells)
        print('  %11s %8s %7s %5s %8s ' % ('bin', 'atoms', '%pure*', 'cells*',
                                          ceiling)
              + ' '.join('%8s' % L for L in cands)
              + ' %17s  best' % 'random')
        for lo, hi in bins:
            sel = bulk & (cerr >= lo) & (cerr < hi)
            n_all = int(sel.sum())
            n_p = int((sel & pure_a).sum())
            n_pc = len(np.unique(row[sel & pure_a]))
            if excl:
                sel = sel & ~pure_a
            n = int(sel.sum())
            lab = ('%g-%g' % (1e3 * lo, 1e3 * hi) if np.isfinite(hi)
                   else '>=%g' % (1e3 * lo))
            if n < E.PERATOM_MIN_BIN_ATOMS:
                continue
            md = {L: 1e3 * E._wquantile(err[L][sel], np.ones(n), 0.5)
                  for L in labels}
            mean, sd, _ = E._seed_stats([md[L] for L in labels], labels)
            vals = {L: md[L] for L in cands}
            if seeds:
                vals['random'] = mean
            ranked = sorted(vals.items(), key=lambda kv: kv[1])
            best = (ranked[0][0] if len(ranked) > 1 and np.isfinite(sd)
                    and ranked[1][1] - ranked[0][1] > sd else '-')
            pct_p = 100.0 * n_p / n_all if n_all else float('nan')
            print('  %11s %8d %6.1f%% %5d %8.1f ' % (lab, n, pct_p, n_pc,
                                                     md[ceiling])
                  + ' '.join('%8.1f' % md[L] for L in cands)
                  + ' %8.1f +- %5.1f  %s' % (mean, sd, best))
            recs.append(dict(selection=('bulk_without_pure_' + pure if excl
                                        else 'bulk_all'),
                             bin_meV_per_A=lab, n_atoms=n,
                             pct_atoms_pure_in_full_bin=pct_p,
                             n_pure_cells_in_full_bin=n_pc,
                             **{'median_' + L: md[L] for L in labels},
                             seed_mean=mean, seed_sd=sd, best=best))

    if not recs:
        raise SystemExit('no bulk bin has enough atoms to report')
    path = os.path.join(out_dir, 'bulk_profile_pure_element_excluded.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print('\n  wrote %s' % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', required=True,
                    help='REQUIRED: per-model *_paper directories '
                         '(per_structure.npz + per_atom_forces.npz)')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='one label per --models entry; rdm*/seed*/random* '
                         '= random-seed replicas (default: directory names)')
    ap.add_argument('--ceiling', required=True,
                    help='REQUIRED: label of the full-data reference model '
                         '(MACE-OMAT-0) whose per-atom error cuts the bins')
    ap.add_argument('--configs', required=True,
                    help='REQUIRED: the MatPES extxyz the models were '
                         'evaluated on')
    ap.add_argument('--threshold', type=float, default=10.0,
                    help='per-atom ||dF_i|| threshold in eV/A, strict >')
    ap.add_argument('--pure-element', default='N',
                    help='element whose single-element cells are singled out')
    ap.add_argument('--non-bulk-detector', default='dim',
                    choices=sorted(E.NON_BULK_CRITERIA) + ['none'],
                    help="bulk/non-bulk criterion re-derived from the stored "
                         "descriptors ('none' = stored split)")
    ap.add_argument('--out-dir', default='.',
                    help='directory for the output CSVs')
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(p))
                             for p in args.models]
    if len(labels) != len(args.models) or len(set(labels)) != len(labels):
        raise SystemExit('--labels must give one distinct label per --models')
    if args.ceiling not in labels:
        raise SystemExit('--ceiling %r is not one of --labels' % args.ceiling)
    from ase.data import chemical_symbols
    if args.pure_element not in chemical_symbols[1:]:
        raise SystemExit('--pure-element %r is not an element symbol'
                         % args.pure_element)
    os.makedirs(args.out_dir, exist_ok=True)

    base, err, row, z = load_models(args.models, labels, args.non_bulk_detector)
    print('models: %s; ceiling: %s; seeds: %s'
          % (', '.join(labels), args.ceiling,
             ', '.join(L for L in labels if E._is_seed_label(L)) or 'none'))
    part_a(base, err, row, z, labels, args.ceiling, args.configs,
           args.threshold, args.pure_element, args.out_dir)
    part_b(base, err, row, labels, args.ceiling, args.pure_element,
           args.out_dir)


if __name__ == '__main__':
    main()
