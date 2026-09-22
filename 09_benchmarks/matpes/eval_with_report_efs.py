#!/usr/bin/env python3

"""
Run a MACE model over an extxyz dataset (mace.cli.eval_configs) and print the
energy, force and stress error MAE/RMSE at the end.

This is eval_with_report.py extended from forces-only to energy + forces +
stress (hence "_efs"). The force numbers it prints are computed exactly as
before, so they are directly comparable with the older script's output.

WHAT v2 ADDS
============
The v1 report is a POOLED MAE/RMSE over every atom of the test set, and it
keeps only scalar sums, so no median, quantile, CDF or paired statistic can be
recovered from it afterwards. On MatPES that pooled statistic cannot resolve a
model comparison in either direction: 1-3 structures out of 323,096 carry 50%
of the total squared force error and 6-37 carry 99%, so four models differing
only in the random seed of their training subset span a factor of 88 in pooled
force MAE and a factor of 307 in pooled RMSE.

v2 therefore records ONE ROW PER STRUCTURE -- errors, geometric descriptors and
reference magnitudes -- prints six additional PAPER TABLE sections from it, and
saves the table so any split, estimator or figure can be produced later without
re-running the model:

  PAPER TABLE 1  ROBUST ACCURACY: median, quantiles, upper-trimmed mean and
                 geometric mean of the per-structure error, per domain, with
                 the pooled mean (the v1 MAE) printed last for reference.
  PAPER TABLE 2  STABILITY: how many structures exceed fixed error thresholds.
                 Reported apart from accuracy, because a model that is
                 typically accurate but occasionally divergent and one that is
                 uniformly mediocre are different failures, and no single
                 average can tell them apart.
  PAPER TABLE 3  ERROR CONCENTRATION: how many structures carry 50/90/99% of
                 the total squared error. Print this beside any pooled RMSE.
  PAPER TABLE 4  CDF of the per-structure force MAE (also written as CSV).
  PAPER TABLE 5  the same metrics split by max |F_ref|, a split computed from
                 the DFT reference alone and therefore model-independent.
  PAPER TABLE 6b the ceiling-conditional profile again, scored per ATOM:
                 the median of ||dF_i|| over every atom of the bin, with no
                 per-structure mean taken first.
  PAPER TABLE 6c the same bins scored as MACE's MAE_F -- the mean of
                 |dF_i,k| over all 3N components of the bin's atoms.
                 6b and 6c need per_atom_forces.npz (written by default) for
                 this model and for the ceiling; without it they are skipped
                 and every other table is unchanged.
  PAPER TABLE 6  with --ceiling, the PAIRED difference and ratio against a
                 CEILING MODEL trained on the full dataset (e.g.
                 MACE-OMAT-0), plus a profile conditioned on that model's own
                 error. Note the terminology: "reference" throughout this
                 script means the DFT ground truth read from the test-set
                 file, never a model. The ceiling model is just another
                 prediction scored against that same DFT reference, so its
                 own error is NOT zero -- on MatPES its median per-structure
                 force MAE is about 100 meV/A.

WHAT v5 ADDS
============
One table, in --aggregate only, and nothing else changes: every number, line
and file v4 produced is produced identically.

  PAPER TABLE 2b BLOW-UP CENSUS: how many ATOMS have a force error whose own
                 NORM ||dF_i|| exceeds a threshold, per model and per domain.

Table 2 (STABILITY) already counts things above a force threshold, but it
counts STRUCTURES and it tests the per-structure, component-averaged MAE. That
average is taken over the structure's own atoms, so one divergent atom in a
216-atom cell is diluted 216-fold while the same atom in a 4-atom cell is not:
the count answers "how many structures came out bad", which is not the same
question as "how many atoms blew up", and the two cannot be converted into
each other after the fact. Quoting a count of atoms from a table of structures
(or from the atom-weighted CDF, which weights each structure by its size but
still thresholds the structure average) is the mistake this table exists to
remove.

2b counts atoms directly. An atom is counted when ||dF_i|| > t, with no
per-structure mean taken first, so the result is a straight census of the
divergent atoms and can be quoted as one. It needs per_atom_forces.npz for
every model; without it the table is skipped and, as with 6b/6c, every other
table is unchanged.

The domains are the usual ALL / BULK / NON-BULK / OVERCOMPRESSED, plus
BULK+NON-BULK -- the union of the two physical domains with the overcompressed
screen excluded, which is the population most figures are drawn on. It is
built from the masks rather than as ALL minus OVERCOMPRESSED so that it stays
correct regardless of whether the domains happen to partition the set.

  PAPER TABLE 1c PER-ATOM FORCE MAE: the pooled per-atom force error of each
                 model on each domain, in BOTH conventions side by side.

Table 1 reports estimators of the per-STRUCTURE error distribution, and its
'pooled_mean_MAE' row is the only pooled per-atom number in it -- available
for the component convention alone, because f_mae_norm is not one of the
tabulated _QUANTITIES. So the report has never carried the norm-convention
per-atom MAE at all, even though the column exists in per_structure.npz.

1c adds both, per domain:

    norm  = mean over the domain's ATOMS of ||dF_i||
    comp  = mean over the domain's 3N COMPONENTS of |dF_i,k|  -- MACE's MAE_F,
            the number a MACE training log prints

Neither is an approximation. For a structure of N_s atoms, f_mae_comp_s is the
mean of |dF_i,k| over its own 3N_s components and f_mae_norm_s the mean of
||dF_i|| over its own N_s atoms, so weighting each by N_s and dividing by the
domain's total gives exactly the mean over every atom (or component) in the
domain. The table therefore needs only per_structure.npz and is produced even
when per_atom_forces.npz is absent. When the per-atom files ARE present the
same two numbers are recomputed from the raw atoms and the two routes are
compared, so a disagreement is reported rather than assumed away.

The two conventions are not interchangeable and no constant converts them:
||dF_i|| <= |dF_i,1| + |dF_i,2| + |dF_i,3| bounds their ratio to [1, 3], and
the value inside that range depends on how anisotropic the error vectors are
(it is exactly 2 only for isotropic Gaussian errors). Quoting one against a
number computed with the other is a mistake the report should make it hard to
commit, which is why they are printed as two rows of one table.

--aggregate reads the per-structure tables of several models and prints the
combined tables, using the random-seed replicas as a measured NOISE FLOOR:
every metric gets the seed mean +- sd, and every model gets its offset from
that mean in units of that sd. A difference smaller than the seed sd is not a
result, whatever its sign, and being able to say so quantitatively is what lets
the pooled MAE and RMSE stay in the paper honestly instead of being quietly
dropped.

Design rules this section follows, and why
------------------------------------------
* Nothing is discarded to make a number look better. Non-finite errors are
  mapped to +inf rather than dropped, exactly as v1 keeps them in its sums, so
  a divergent model ranks last instead of being improved by the exclusion.
* No domain is defined from any model's errors, the ceiling model's included.
  Selecting structures on a noisy quantity biases that model's own numbers
  inside the selection, and it removes precisely the structures where some
  candidate fails. Every split offered here is computed from the reference
  structure or the reference forces. The one table that does bin on the
  ceiling's error prints every bin and states that only the candidate models
  may be compared within a bin.
* The per-structure table is a lossless refinement of v1: the atom-weighted
  pooled mean of the per-structure force MAE reproduces the v1 component-wise
  MAE exactly, per domain. That identity is the check that the two layers have
  not drifted apart.
* The v1 report is untouched. With collection enabled, the leading lines of
  output are byte-identical to what v1 prints on the same inputs.

The random-subsampling machinery (FRACTION) is inherited from v1 and must stay
at 1.0 for anything reported: every cross-model number here is paired per
structure, and the pooled RMSE does not subsample at all. A run with
FRACTION != 1.0 prints a warning in the paper section, and --aggregate refuses
to combine tables that do not cover exactly the same structures.

Why the metrics are not computed from the eval output alone
-----------------------------------------------------------
mace.cli.eval_configs writes MACE_energy / MACE_forces / MACE_stress into each
frame, but it does `atoms.calc = None` first. Under ASE's extxyz reader the
*reference* `forces`, `energy` and `stress` of the input file live in a
SinglePointCalculator (they are moved out of atoms.info/atoms.arrays on read),
so clearing the calculator drops them: the output file holds ONLY the
predictions -- there is no REF_forces column in it. The reference values
therefore have to be re-read from the input configs file. Frame order is
preserved by eval_configs (DataLoader with shuffle=False, drop_last=False), so
the two files can be zipped frame by frame.

Both files are streamed with ase.io.iread and only scalar sums are kept, so the
reporting pass is memory-bounded regardless of dataset size (the eval itself
still loads the dataset in full, as it always did).

Stress requires --compute_stress
--------------------------------
eval_configs writes MACE_stress ONLY when it is called with --compute_stress;
without that flag the key is simply absent from the output file. The older
forces-only script never passed it, so its output files carry no stress and
cannot be re-reported with --skip_eval here. This script passes the flag by
default; use --no_compute_stress for a model whose head cannot produce stress.

How the reference and the prediction are stored (verified by round-trip)
-----------------------------------------------------------------------
  reference   energy  -> atoms.calc.results['energy']            scalar, eV
              forces  -> atoms.calc.results['forces']            (N,3), eV/A
              stress  -> atoms.calc.results['stress']            Voigt-6, eV/A^3
                         (ASE reshapes the 9 numbers in the comment line to
                          3x3 and then converts to Voigt on read)
  prediction  MACE_energy -> atoms.info, scalar
              MACE_forces -> atoms.arrays, (N,3)
              MACE_stress -> atoms.info, full 3x3 (written by eval_configs as a
                          _JSON nested list; 'MACE_stress' is not one of ASE's
                          special 3x3 keys, so it is read back as-is)
Both sign conventions agree: MACE's stress is +(1/V) dE/dstrain, the same as
ASE's, and MACE trains against atoms.get_stress() directly, so the reference
and the prediction can be subtracted without any conversion.

Files that keep the reference under REF_energy / REF_forces / REF_stress
instead of in the calculator are handled too, as are stresses stored as a
9-vector or as a Voigt-6 vector (Voigt order xx, yy, zz, yz, xz, xy -- ASE's
order, which is also what mace.tools.torch_tools.voigt_to_matrix assumes).

Error conventions
-----------------
For each quantity two conventions are printed, the same idea as
analyze_prediction_v2_fixed_spin_twocutoffs_opt_dists_and_A.py:

  forces  * norm-based:     per atom, e_i = ||F_pred,i - F_ref,i||   (N values)
          * component-wise: per force component (3N values) -- the definition
            used in the MACE training/eval logs (RMSE_F), so compare against
            that one.

  stress  * norm-based:     per structure, e = ||S_pred - S_ref||_F, the
            Frobenius norm of the 3x3 difference  (n_struct values)
          * component-wise: over all 9 components of the 3x3 tensor
            (9*n_struct values) -- this is exactly MACE's mae_stress /
            rmse_stress (it stores stress as [1,3,3] and averages over every
            element), so compare RMSE_stress against that one.
          A Voigt-6 component-wise variant is printed as well, because parts of
          the literature average over the 6 independent components instead; it
          differs from the 9-component number because the 3 off-diagonals are
          counted once rather than twice.

  energy  energy is a scalar, so there is no component decomposition and the
          two conventions collapse to a choice of normalisation:
          * per structure: e = |E_pred - E_ref|            (n_struct values)
          * per atom:      e = |E_pred - E_ref| / N_atoms  -- the MACE log
            convention (RMSE_E_per_atom), so compare against that one.

Consistency relations worth knowing when reading the output (they are exact,
not approximations, and are a good sanity check that nothing is misaligned):
  RMSE_forces_norm = sqrt(3) * RMSE_forces_component
  RMSE_stress_norm = 3       * RMSE_stress_component_9
The MAEs are NOT related by any such factor.

Non-finite values are counted and reported but NOT dropped
---------------------------------------------------------
A divergent model can emit inf/nan. Silently skipping those frames would make
such a model look better than it is, so they stay in the sums: the affected
metric comes out as nan and a WARNING line tells you how many frames caused it.
Frames with no cell (or a zero-volume cell) are excluded from the stress
statistics only -- MACE clamps stress to 0 there, which is not a prediction.

Random subsampling (FRACTION / SEED, set at the top of this file)
----------------------------------------------------------------
With FRACTION F (0 < F <= 1) only a random subset of the frames is evaluated.
The selected frames are copied VERBATIM (raw byte ranges, no parse/reformat
round-trip) into a subset extxyz file, the model is run on that file, and the
metrics compare the subset against its own predictions.

The draw depends ONLY on (SEED, FRACTION, number of frames in the reference),
never on the model, the device or the dtype. So running two different models
with the same seed, fraction and reference file evaluates exactly the same
structures and the numbers are directly comparable. Changing the reference file
(or its frame count) changes the draw -- that is intended, but it means a
comparison is only valid within one reference file.

CAVEAT -- do not quote a subsampled RMSE. The force RMSE on this test set is
dominated by a handful of frames (~0.1% of frames carried >80% of the total
squared error), so a draw either catches them or does not. Measured over 300
independent 10% draws of the 323,096-frame MatPES set:

                       full set      10% draw (median, p5-p95)      spread
  MACE-OMAT-0   MAE      342.68      341.48  [ 319.8 -    377.2]     17%
                RMSE    1357.87      977.69  [ 801.8 -   2298.7]    153%
  subset model  MAE     8555.91     6665.97  [3687.1 -  18044.8]    215%
  (divergent)   RMSE 1228562.00   731763.90  [329980 - 2461633]     291%
                                                        (meV/A, norm-based)

So: for a well-behaved model a 10% MAE is trustworthy (median within 0.3% of
the full-set value), while its RMSE is biased ~28% low with a 153% spread. For
a model with a divergent tail nothing subsamples -- even the MAE swings by a
factor of 5. Use FRACTION for quick MAE comparisons and pipeline smoke tests
only, and run the full set for anything you report.

That caveat was measured on forces, but it applies at least as strongly to the
energy and stress RMSEs: those are per-structure quantities, so a 10% draw
holds 10x fewer samples than the force statistics do and the tail is sampled
correspondingly worse.

Side benefit: if the subsampled MAE moves a lot when you change only SEED,
that instability is itself evidence of a heavy-tailed / divergent model.

Force-error cutoffs (FORCE_ERROR_CUTOFFS, FORCE_ERROR_CUTOFF_ON)
----------------------------------------------------------------
On top of the report over everything, the same metrics can be reported again
with the worst atoms taken out. FORCE_ERROR_CUTOFFS is a list of thresholds in
eV/A; for each one an extra block is printed.

MACE's force error is not a per-atom quantity. It pools every Cartesian
component of every atom of every structure into one flat array of 3N scalars
and averages that:

    MAE_F  = mean( |dF_i,k| )        over all 3N components
    RMSE_F = sqrt( mean( dF_i,k^2 )) over all 3N components

A cutoff, however, needs one number per atom, so an atom's three components
have to be reduced to one -- a step MACE itself never takes.
FORCE_ERROR_CUTOFF_ON chooses that reduction:

  'rms_component' (default)  the atom's own RMS over its three components,
                             sqrt( (1/3) * sum_k dF_i,k^2 ). This is the
                             per-atom form of the component-wise error, on the
                             same scale as the RMSE_F printed in a MACE log, so
                             a cutoff of n is directly readable against that
                             number.

  'max_component'            max_k |dF_i,k|: an atom is cut when any SINGLE
                             component exceeds the threshold.

  'norm'                     ||dF_i||, the error vector length.

The atom stays the unit of exclusion in every case. Individual components are
never dropped on their own: the norm-based force metric is undefined for an
atom with only some of its components kept, so a component-level filter would
make half this report incoherent.

'rms_component' is exactly ||dF_i|| / sqrt(3) -- the constant sqrt(1/3) factors
straight out of the sum under the square root. So it orders atoms identically to
'norm' and selects the identical set of atoms that 'norm' would at a threshold
sqrt(3) ~ 1.732 times larger:

    rms_component > c   <=>   norm > sqrt(3) * c

Those two are therefore the same criterion in different units, and switching
between them only rescales the threshold. 'max_component' is a genuinely
different ordering: since max_k |dF_i,k| <= ||dF_i|| <= sqrt(3) * max_k
|dF_i,k|, it can keep an atom whose error is spread evenly over all three
directions while cutting one whose error sits in a single direction. The report
header states which criterion ran, and any number you quote should say so too.

The cutoff is a per-atom test, but energy and stress are per-structure
quantities and cannot be partially excluded, so each cutoff is reported two
ways:

  [a] whole structures dropped
      A structure is kept only if EVERY one of its atoms is at or below the
      cutoff. Energy, forces and stress are then all computed over exactly the
      same surviving structures, using all of their atoms. This answers "how
      good is the model on the configurations it did not blow up on?".

  [b] offending atoms dropped, all structures kept
      Only the atoms above the cutoff leave the force statistics. Energy and
      stress are computed over every structure, so those rows are identical to
      the uncut block by construction and are marked as such. This answers "how
      good are the forces once the outlier atoms are removed?".

The force numbers therefore DIFFER between [a] and [b]: [a] keeps every atom of
a clean structure and discards clean atoms that merely share a structure with a
bad one, while [b] keeps clean atoms from every structure. Comparing the two is
the informative part -- if [a] and [b] agree, the bad atoms are scattered; if
[a] is much better, the errors are concentrated in a few whole structures.

Each block also reports how much was removed (atom and structure counts and
percentages) and what share of the total squared force error the excluded atoms
carried. That share is the sum of ||dF_i||^2 over the excluded atoms divided by
the same sum over all atoms -- identical whether you think of it per component
or per atom, since ||dF_i||^2 is exactly the sum of its three squared
components, and so independent of which criterion selected the atoms. Always read the metrics together with those numbers: a cutoff that
removes 30% of the atoms is not measuring the same thing as one that removes
0.01%, and quoting a cut MAE without saying what was cut is meaningless.

Every cutoff block is broken down over the same domains as the uncut DOMAIN
SPLIT section: ALL DATA first, then BULK, NON-BULK and OVERCOMPRESSED. Both
variants [a] and [b] are printed for each domain, so a cut number can be read
on the domain it belongs to instead of only on the pooled set. Domain
membership is decided on the reference structure alone, so it is identical in
every cutoff block; the cutoff only decides which atoms and structures inside a
domain are counted. The domains partition the test set, so ALL DATA is the
union of the others and nothing is double-counted.

Inside a domain block every count and percentage is taken over that domain's
own totals -- its own atom count, its own structure count, its own summed
squared force error -- never over the full set. A cutoff that removes 0.01% of
BULK can therefore remove several percent of NON-BULK, and the two numbers are
meant to be read against their own denominators.

Set FORCE_ERROR_CUTOFFS = [] (or pass --force_error_cutoffs with no values) for
the uncut report only.
"""

import argparse
import os
import sys
from collections import deque
from itertools import zip_longest

import ase.io
import numpy as np

from mace.cli.eval_configs import main as mace_eval_configs_main


# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK (or pass the flags) <<<
#
# TEST_SET_PATH / --configs
#     REQUIRED. The MatPES test set as a single extxyz FILE, carrying DFT
#     energy / forces / stress. The published run used MatPES PBE 2025.2 with
#     rare-earth structures removed (323,096 structures / 3,011,678 atoms) --
#     the SAME file stage 06 uses as its test_file. Not produced by this
#     repository; see DATA.md for where to get it and what to filter.
#     None here means "must be given on the command line".
#
# MODEL_PATH / --model
#     REQUIRED. The MACE model FILE to evaluate. Use the *_compiled.model
#     exported by stage 08 (MatPES and MatCalc take the compiled export; the
#     stability and NEB benchmarks take *_run-1.model instead).
#
# OUT_PATH / --output
#     Output extxyz of per-structure predictions. A sibling directory
#     `<OUT_PATH without extension>_paper/` is also written, and that directory
#     is what --aggregate consumes.
#
# --label
#     How this model is named in the aggregate tables. IT MATTERS: --aggregate
#     treats any label starting with rdm / seed / random as a random-seed
#     REPLICA defining the noise floor. Label replicas rdm1..rdm4 and everything
#     else descriptively (Chemenv, OMAT0, DIRECT, ...).
#
# The statistical settings below (SEED, FRACTION, cutoffs, domain split) change
# what is reported. Leave them as they are to reproduce the paper's tables.
# ============================================================================
TEST_SET_PATH = None                        # REQUIRED -- give it with --configs
MODEL_PATH = None                           # REQUIRED -- give it with --model
OUT_PATH = 'test_matpes_efs.xyz'            # override with --output
# ============================================================================

# MERGED (repo change): the as-run code shipped two 193 KB copies of this file,
# one per evaluated model, differing ONLY in MODEL_PATH / MODEL_LABEL / OUT_PATH
# (`diff` was 9 lines). Both values were already exposed as --model / --output /
# --label, so the copies are collapsed into this single file and the submit
# scripts pass the flags. The duplicate `MODEL_LABEL = 'Chemenv'` that used to
# sit here was dead code -- it is reassigned to None further down (see the
# SEED_LABEL_PREFIXES note) and the second assignment always won.

# --- random subsampling ----------------------------------------------------
# FRACTION: 1.0 evaluates the whole dataset. 0.1 draws a random 10% of the
#           frames, which makes the evaluation ~10x cheaper.
FRACTION = 1.0

# SEED:     the draw depends ONLY on (SEED, FRACTION, number of frames in
#           TEST_SET_PATH) -- never on the model, device or dtype. Keep these
#           three fixed and every model you evaluate sees exactly the same
#           structures, so the numbers are directly comparable.
SEED = 42

# SUBSET_FILE: where the sampled reference frames are written.
#              None -> '<OUT_PATH without extension>_subset.xyz'
SUBSET_FILE = None

# KEEP_SUBSET: the subset file is exactly reproducible from the three settings
#              above, so it is deleted after the run unless this is True.
KEEP_SUBSET = False

# Volume below which a cell counts as "no cell" and the frame is left out of
# the stress statistics (MACE zeroes stress there instead of predicting it).
MIN_CELL_VOLUME = 1e-8

# --- force-error cutoffs ---------------------------------------------------
# Extra report blocks with the worst atoms excluded, one block per threshold,
# in eV/A. [] -> only the report over everything. See the module docstring for
# what the [a] / [b] variants of each block mean.
FORCE_ERROR_CUTOFFS = [10., 20., 30.]

# Which per-atom statistic the cutoff is applied to:
#   'rms_component' -> sqrt(mean_k (dF_i,k)^2) > cutoff
#                      The atom's own RMS over its 3 error components, i.e. the
#                      per-atom version of the component-wise error MACE
#                      averages for RMSE_F, on the same scale as the RMSE_F
#                      number in a MACE log. Identically equal to
#                      ||dF_i|| / sqrt(3) -- see the note below.
#   'max_component' -> max_k |dF_i,k| > cutoff  (an atom is cut if ANY single
#                      component exceeds the threshold)
#   'norm'          -> ||dF_i|| > cutoff        (error vector length)
FORCE_ERROR_CUTOFF_ON = 'rms_component'

# 'rms_component' is computed straight from the components, per its definition.
# It is EXACTLY the norm-based per-atom error divided by sqrt(3):
#     sqrt( (1/3) * sum_k d_k^2 )  ==  sqrt( sum_k d_k^2 ) / sqrt(3)
#                                  ==  ||d|| / sqrt(3)
# because sqrt(1/3) factors out of the sum. So it ranks atoms in exactly the
# same order as 'norm' and selects exactly the same atoms as 'norm' would at a
# threshold sqrt(3) times larger. It is a rescaling, not a different ordering.
# The two differ only in what a given threshold NUMBER means:
#     rms_component > c   <=>   norm > sqrt(3) * c
RMS_COMPONENT_TO_NORM = 3.0 ** 0.5    # ||dF_i|| = sqrt(3) * rms_component_i

CUTOFF_CRITERIA = {
    'rms_component': (
        'sqrt(mean_k (F_pred,i,k - F_ref,i,k)^2)  (per-atom RMS over the 3 '
        'components; == ||dF_i||/sqrt(3))',
        lambda f_diff, f_norms: np.sqrt(np.square(f_diff).mean(axis=1))),
    'max_component': ('max_k |F_pred,i,k - F_ref,i,k|  (largest component)',
                      lambda f_diff, f_norms: np.abs(f_diff).max(axis=1)),
    'norm': ('||F_pred,i - F_ref,i||  (error vector length)',
             lambda f_diff, f_norms: f_norms),
}


# --- non-bulk (vacuum-containing) structure detection ----------------------
# Reports the same metrics split into a "bulk" and a "non-bulk" domain instead
# of discarding anything. None disables the split entirely.
#
#   'dim'          periodic dimensionality < 3. A structure is bulk only if
#                  its bonded network extends in all three directions; a slab
#                  is 2-periodic, a chain 1-periodic, an isolated cluster
#                  0-periodic. Computed Larsen-style: for each bonded fragment,
#                  take the lattice offsets of the bonds that leave the home
#                  cell and use the RANK of that set of offsets.
#                  This is the recommended criterion because it is TOPOLOGICAL:
#                  a slab is 2-periodic whether its vacuum gap is 4 A or 40 A,
#                  so there is no magnitude to justify. Its one parameter is
#                  NON_BULK_BOND_SCALE, and the MatPES classification does not
#                  change over 1.15-1.40. Costs ~5 ms per structure.
#                  ON ITS OWN IT IS NOT ENOUGH: a 3D framework holding a free
#                  molecule in a pore is still 3-periodic, so 'dim' misses
#                  exactly the vacuum-isolated clusters you care about. On the
#                  MatPES subset, retaining dim==3 left 14 diverging structures
#                  and a v14 RMSE of 104,823 meV/A. Use 'dim_or_frag'.
#   'dim_or_frag' (default)  dimensionality < 3 OR more than one connected
#                  fragment. This is the recommended criterion: the first test
#                  catches slabs, layers and chains, the second catches
#                  clusters and molecules sitting in vacuum -- including inside
#                  a 3D host, which is the case 'dim' alone cannot see.
#                  Topological, so no magnitude to defend; one parameter
#   'packing'      density index < NON_BULK_PACKING_THRESHOLD (see below).
#                  Needs no neighbour list at all, so ~17 us per structure. Use
#                  it as a cheap screen; see the caveats before quoting it.
#   'fragmented'   more than one connected component in the cell.
#   'min_cn'       at least one atom with coordination number <= 2. This is the
#                  descriptor the atom-level failure analysis points at, but it
#                  flags a quarter of the set, so it diagnoses rather than
#                  defines a domain.
#   'any'          union of 'packing', 'dim_or_frag' and 'min_cn'.
#
# Everything except 'packing' builds a neighbour list per structure, which costs
# a few milliseconds each -- small against the GPU evaluation of the same frames.
NON_BULK_DETECTOR = 'dim_or_frag'

# --- the 'packing' density index, and why it is only a screen --------------
# NOT a packing fraction in the crystallographic sense. It is
#
#       sum_i (4/3) pi R_i^3   /   |det(cell)|
#
# with R from ase.data.covalent_radii. Covalent radii put bonded atoms at
# R_i + R_j, so the spheres touch along bonds and OVERLAP in metals, where
# covalent radii underestimate the separation. The index therefore routinely
# exceeds the 0.74 hard-sphere maximum -- W bcc gives 1.129, Mo 0.979, CsCl
# 0.934 -- and its absolute value is meaningless outside this one radius table
# (van der Waals radii give Si diamond 1.938, and are undefined for 43% of
# elements 1-95).
#
# THIS INDEX CANNOT CLASSIFY BULK vs NON-BULK, and no threshold fixes that.
# It conflates two independent things: how intrinsically open the crystal is,
# and whether the cell contains vacuum. Across materials the first effect is
# larger than the second, so the two populations overlap outright:
#
#   material      bulk    +4A vac  +8A vac  +15A vac
#   W bcc        1.129     0.691    0.498    0.335   <-- slab beats bulk Si
#   Cu fcc       0.819     0.527    0.389    0.266
#   Al fcc       0.447     0.299    0.225    0.157
#   Si diamond   0.286     0.209    0.165    0.120   <-- lowest bulk tested
#
# A tungsten slab with a 15 A vacuum gap (0.335) scores HIGHER than perfect
# bulk silicon (0.286), because Si is 4-coordinate and open while W is
# 8-coordinate and dense. Within one material the index does fall monotonically
# with vacuum -- it is the comparison across materials that breaks.
#
# On the MatPES subset, against the topological definition of non-bulk:
#   bulk (dim==3, 1 fragment)  median 0.572   p5 0.260   max 1.890
#   dim==2 slab/layer          median 0.230              max 0.782
#   dim==1 chain               median 0.083              max 0.413
#   dim==0 cluster             median 0.064              max 0.374
#   ROC AUC 0.928; best accuracy 0.894 at a 0.25 threshold, where recall is
#   only 0.622 -- it misses 2,234 of 5,912 non-bulk structures and falsely
#   flags 1,181 bulk ones.
# There is also no natural break in the distribution near 0.25 (it rises
# smoothly through it; the only local minimum is near 0.11).
#
# Why it still scores well on blow-up recall (96.2%, better than dim_or_frag's
# 94.2%): the divergences live in the EXTREME low tail -- an isolated molecule
# in a large vacuum gives 0.001-0.1, which any threshold catches. The non-bulk
# structures it misses are mild layered cases that do not diverge. So the index
# is a usable detector of SEVERE vacuum and a poor classifier of NON-BULK.
# The two screens catch different failures: 'dim<3' flags slabs, wires and
# molecules, 'fragmented' flags cells that fall into disconnected pieces. Either
# alone misses cases the other catches, so the default uses both.
NON_BULK_PACKING_THRESHOLD = 0.25

# Bond cutoff for the graph-based detectors ('dim', 'fragmented',
# 'dim_or_frag', 'min_cn', 'any'): a pair counts as bonded when
#       d_ij < NON_BULK_BOND_SCALE * (R_cov,i + R_cov,j).
# It is NOT read by the 'packing' detector, which touches no neighbour list.
# The MatPES classification was stable over 1.15-1.40, which is the robustness
# check to quote if you publish a dimensionality-based domain.
NON_BULK_BOND_SCALE = 1.25

# Calibration on the 32,310-frame MatPES PBE subset, with "diverges" meaning a
# structure carrying an atom above 30 eV/A of force error:
#
#   packing<thr   flagged   v14 recall   DIRECT recall   share of v14 sum e^2
#      0.10         5.1%       73.1%         86.9%              98.420%
#      0.15         8.1%       88.5%         91.9%              99.955%
#      0.20        11.3%       94.2%         97.0%              99.966%
#      0.25        15.0%       96.2%         98.0%              99.999%
#      0.30-0.40   20-34%      96.2%         98.0%              99.999%
#
# Recall plateaus above ~0.20, so anything in 0.20-0.40 gives the same answer;
# the threshold is not a tuned parameter. Volume-per-atom was tested as an
# alternative and is strictly worse (94.2% recall costs 23.4% of the set
# flagged, against 11.3% for packing).
#
# --- overcompressed / unphysical-contact screening -------------------------
# Structures with a pair of atoms far closer than a bond are a data-quality
# problem rather than a domain choice: the DFT reference forces there run to
# hundreds of eV/A and every model tested fails on them, MACE-OMAT-0 included.
# They are reported as their own domain so they never silently contaminate
# either the bulk or the non-bulk numbers. None disables the screen.
#
# CONTACT RATIO, exactly:
#     min over all pairs (i, j) and all periodic image offsets o, excluding
#     only the i == j / o = 0 self-pair, of
#             | r_i - (r_j + o . A) |  /  ( R_i + R_j )
#     where A is the 3x3 lattice matrix and R is the element's covalent radius
# So 1.0 means the closest pair sits exactly at the sum of its covalent radii,
# and 0.55 means some pair is at 55% of that distance. An atom's own periodic
# images are included, so a cell collapsed smaller than one atom is caught too.
#
# On the MatPES subset 15 of 32,310 structures fall below 0.55, and just two of
# them carry 95% (v14), 94% (DIRECT) and 86% (MACE-OMAT-0) of all the squared
# force error that survives in the dense half of the set.
CONTACT_RATIO_CUTOFF = 0.4

# KNOWN BLIND SPOT of the packing detector: it detects vacuum, not compression. A cell with an
# unphysically short bond is DENSE, so a high packing fraction, and will land
# in the "bulk" half however low you set the threshold. On the MatPES subset
# those cells are what leaves an RMSE of ~2500 meV/A in the bulk half for all
# three models tested, MACE-OMAT-0 included. That is what CONTACT_RATIO_CUTOFF
# above is for; the two screens are complementary and both are needed.


# ===========================================================================
# v2 additions: per-structure data capture and the paper tables
# ===========================================================================
# The v1 report keeps only scalar sums, so no median, quantile, CDF or paired
# statistic can be recovered from it. v2 additionally records ONE ROW PER
# STRUCTURE (errors + geometric descriptors + reference magnitudes), prints the
# paper tables from it, and saves it so plots can be made without re-running
# the model. 323,096 structures x ~35 columns is ~90 MB in memory, so this is
# affordable at MatPES scale; set COLLECT_PER_STRUCTURE = False to restore the
# v1 memory profile exactly.
COLLECT_PER_STRUCTURE = True

# Short name for this model in the saved tables and in --aggregate. Set it
# here next to MODEL_PATH / OUT_PATH, or override it with --label.
#
# IT MATTERS WHAT YOU CALL THE RANDOM-SEED REPLICAS. --aggregate treats a
# label beginning with rdm / seed / random (see SEED_LABEL_PREFIXES) as one of
# the replicas that define the noise floor, and that noise floor is what every
# significance statement in the paper tables rests on. Label them rdm1..rdm4.
# None -> derived from MODEL_PATH's filename, which for these models does NOT
# begin with 'rdm' and would silently leave you with no noise floor at all.
MODEL_LABEL = None

# are written. None -> '<OUT_PATH without extension>_paper'. Keep it next to
# the model: --aggregate takes these directories by name, so there is no need
# to collect them anywhere central.
PAPER_OUTPUT_DIR = None

# Also dump every per-atom force error (3,011,678 rows for MatPES, ~50 MB) as
# float32. Needed only for a per-atom error-distribution figure.
SAVE_PER_ATOM = False

# v2.1 ADDITION. Write per_atom_forces.npz beside per_structure.npz: the raw,
# UNBINNED per-atom force error, one row per atom, so any per-atom statistic
# an arbitrary set of atoms) can be computed offline without re-running the
# model. Costs about 26 bytes per atom -- ~78 MB for MatPES's 3.01 M atoms.
# Turn off with --no_per_atom_forces. This is independent of SAVE_PER_ATOM,
# which keeps writing its own per_atom.npz exactly as before.
SAVE_PER_ATOM_FORCES = True

# The exact minimum contact ratio of every structure, not just of the ones
# below CONTACT_RATIO_CUTOFF. Costs one extra neighbour search per frame
# (~2 ms), i.e. ~10 min over MatPES. Worth it: it turns the overcompression
# screen into a continuous descriptor you can re-threshold offline.
SAVE_TRUE_CONTACT_RATIO = True

# --- paper table parameters ------------------------------------------------
# Upper-trim fraction for the trimmed mean. 0.05 = drop the worst 5% of the
# weight. A trimmed mean is reported alongside the median because the median
# alone can hide a model that is catastrophic on a physically relevant
# minority (Morrow, Gardner & Deringer, arXiv:2211.12484).
#
# READ THIS BEFORE QUOTING A TRIMMED MEAN ACROSS MODELS. Trimming a fixed
# FRACTION removes a different AMOUNT of error from each model, and it removes
# more from the models with the heaviest tails -- so it flatters exactly the
# behaviour it looks like it is guarding against. Trimming removes a fixed
# FRACTION, so it removes far more total error from a model with a few extreme
# frames than from one without, and can therefore rank the model with the heavier
# tail as the better one. Where models differ mainly in their tails, the trimmed
# mean and the median can disagree, and the trimmed mean is the one to distrust.
#
# So: the trimmed mean is safe for comparing DISTRIBUTION SHAPE within one
# model, and unsafe as a cross-model comparator whenever tail shapes differ.
# For cross-model work use the median, or the winsorised mean below, which
# clips every model at the SAME absolute value and therefore treats them
# alike.
TRIM_FRACTION = 0.05

# Quantile of the POOLED values across all models (computed once per quantity
# and domain in --aggregate) at which every model is clipped for the
# winsorised mean. One common absolute cap for all models is what makes it a
# fair comparator; it is unit-agnostic, so the same rule works for energies,
# forces and stresses.
WINSOR_QUANTILE = 0.99

# Quantiles of the per-structure error distribution to tabulate.
PAPER_QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

# Per-structure force MAE thresholds for the STABILITY table, eV/A. These are
# deliberately separate from FORCE_ERROR_CUTOFFS: those cut atoms out of the
# metrics, these only count structures.
STABILITY_FORCE_THRESHOLDS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

# Per-atom energy-error thresholds for the STABILITY table, eV/atom.
STABILITY_ENERGY_THRESHOLDS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5)

# Grid for the cumulative-distribution table and the plot CSVs, eV/A.
CDF_FORCE_GRID = (0.005, 0.010, 0.015, 0.025, 0.050, 0.075, 0.100, 0.150,
                  0.200, 0.300, 0.400, 0.500, 0.750, 1.000, 1.500, 2.000,
                  3.000, 5.000, 10.000)

# Bin edges for the reference-force-magnitude split, eV/A. Model-independent:
# computed from the DFT reference alone, so it cannot flatter any model.
REF_FORCE_BINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)

# Bin edges for the ceiling-conditional profile, eV/A, applied to the
# REFERENCE MODEL's own per-structure force MAE.
CEILING_BINS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)

# v4 ADDITION. Bin edges for the two PER-ATOM-BINNED ceiling profiles.
#
# The v2.1 per-atom tables still cut their bins on the ceiling's PER-STRUCTURE
# f_mae_comp and only the statistic inside a bin is per-atom. These two put the
# binning on the same footing as the statistic: a bin holds the ATOMS whose
# ceiling error falls between the edges, not the atoms of the structures whose
# ceiling MEAN does.
#
# PERATOM_CEILING_BINS is deliberately the same ladder as CEILING_BINS, in the
# same units (eV/A), so the absolute per-atom rows can be read against the
# per-structure rows above them. Per-atom norms sit higher than a structure's
# component-average MAE, so the mass moves up the ladder; that is the effect
# being shown, not a reason to re-cut the bins.
PERATOM_CEILING_BINS = CEILING_BINS

# The relative profile is dimensionless, so it needs its own ladder. It is
# CEILING_BINS itself, read as a FRACTION instead of in eV/A, continued two
# rungs further. The rule is deliberately one a reader can check against the
# table above without reference to any result: same start, same doubling, more
# rungs.
#
# The extra rungs are not cosmetic. Relative errors have a much heavier upper
# tail than absolute ones, because dividing by a small reference force turns a
# modest error into a large ratio. Stopping where the absolute ladder stops
# piles that tail into the open top bin and HIDES it; carrying the ladder two
# rungs further lets it fall off the way the absolute table's does, and a
# sparsely populated top bin is informative -- it is the statement that
# catastrophic relative errors are rare.
#
# rho = 1 -- "the error is at least as large as the force itself" -- is a real
# property worth knowing, and it is recoverable here: it sits inside the
# 0.8-1.6 bin. Anchoring the edges on it instead would make them look chosen
# rather than inherited, which is the one thing a binning choice must not look
# like.
PERATOM_REL_BINS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)

# Atoms with ||F_ref,i|| <= this are dropped from the RELATIVE table only; the
# absolute tables are untouched by it.
#
# 0.0 keeps every atom whose reference force is not exactly zero. It still
# drops a small share of them, whose reference force vanishes IDENTICALLY by
# symmetry (high-symmetry Wyckoff sites): rho is not a large number there, it
# is undefined, and no cutoff can rescue them. The remaining small forces are
# harmless -- the smallest non-zero ||F_ref|| still gives a finite rho in
# float64, and every statistic reported here is a median. A cutoff large
# enough to discard a sizeable fraction of the atoms does change rankings, so
# it would be a different measurement, not a cleaner one. Keep it at 0 unless
# you have a reason, and state the value if you move it.
REL_MIN_REF_FORCE = 0.0

# A per-atom bin below this many ATOMS is omitted from the two tables above,
# mirroring the 20-STRUCTURE floor the per-structure profile uses.
PERATOM_MIN_BIN_ATOMS = 20

# v5 ADDITION. Per-ATOM force-error thresholds for the BLOW-UP CENSUS, eV/A,
# tested against the error NORM ||dF_i||. The ladder is deliberately the same
# as STABILITY_FORCE_THRESHOLDS so that the two tables can be read side by
# side -- that one counts structures whose component-averaged MAE exceeds the
# threshold, this one counts atoms whose own norm does -- but the two are
# separate constants because they threshold different quantities and there is
# no reason a change to one should silently move the other.
PERATOM_BLOWUP_THRESHOLDS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

# Labels treated as the random-seed replicas when aggregating, for the
# mean +- sd noise floor. Matched as a prefix, case-insensitively.
SEED_LABEL_PREFIXES = ('rdm', 'seed', 'random')

# --- pooled per-atom error distribution ------------------------------------
# Table 1's per-structure statistics first take a MEAN over each structure's
# atoms, which is not robust WITHIN a structure: one bad atom inflates that
# whole structure's value. The pooled per-atom distribution has no inner
# average at all and is robust at both levels, so it is reported beside it as
# a robustness check. Keeping every per-atom error would cost 3,011,678 floats
# per model; a log-spaced histogram gives the same quantiles to 0.12% and
# costs ~2 MB, so it is always on.
PERATOM_HIST_BINS = 16000
PERATOM_HIST_LO = -4.0          # log10(meV/A)
PERATOM_HIST_HI = 4.0


def eval_mace(configs, model, output, device='cuda', default_dtype='float32',
              batch_size=8, compute_stress=True):
    sys.argv = ["program", "--configs", configs, "--model", model,
                "--output", output, "--device", device,
                "--default_dtype", default_dtype,
                "--batch_size", str(batch_size)]
    if compute_stress:
        # store_true flag: without it eval_configs writes no MACE_stress at all.
        sys.argv.append("--compute_stress")
    mace_eval_configs_main()


# ---------------------------------------------------------------------------
# Reading reference / predicted quantities out of a frame
# ---------------------------------------------------------------------------
def stress_to_3x3(stress):
    """
    Normalise any of the stress layouts found in the wild to a full 3x3.

    Accepts (3,3), a flat 9-vector (row-major, as eval_configs' 3x3 flattens)
    and a Voigt-6 vector in ASE order (xx, yy, zz, yz, xz, xy) -- the same
    order mace.tools.torch_tools.voigt_to_matrix assumes, so a Voigt reference
    and a 3x3 prediction end up on the same footing.
    """
    s = np.asarray(stress, dtype=np.float64)
    if s.shape == (3, 3):
        return s
    if s.shape == (9,):
        return s.reshape(3, 3)
    if s.shape == (6,):
        xx, yy, zz, yz, xz, xy = s
        return np.array([[xx, xy, xz],
                         [xy, yy, yz],
                         [xz, yz, zz]], dtype=np.float64)
    raise ValueError(
        f'Cannot interpret a stress of shape {s.shape}; expected (3,3), (9,) '
        f'or Voigt (6,).')


def get_ref_forces(atoms):
    """Reference forces of an input frame, from wherever ASE put them."""
    if atoms.calc is not None and 'forces' in atoms.calc.results:
        return np.asarray(atoms.calc.results['forces'], dtype=np.float64)
    for key in ('REF_forces', 'forces'):      # already-merged / non-standard files
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=np.float64)
    return None


def get_pred_forces(atoms, prefix='MACE_'):
    """Predicted forces of an output frame."""
    key = prefix + 'forces'
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key], dtype=np.float64)
    if atoms.calc is not None and 'forces' in atoms.calc.results:
        return np.asarray(atoms.calc.results['forces'], dtype=np.float64)
    return None


def get_ref_energy(atoms):
    """Reference total energy of an input frame, in eV."""
    if atoms.calc is not None and 'energy' in atoms.calc.results:
        return float(atoms.calc.results['energy'])
    for key in ('REF_energy', 'energy'):
        if key in atoms.info:
            return float(atoms.info[key])
    return None


def get_pred_energy(atoms, prefix='MACE_'):
    """Predicted total energy of an output frame, in eV."""
    key = prefix + 'energy'
    if key in atoms.info:
        return float(atoms.info[key])
    if atoms.calc is not None and 'energy' in atoms.calc.results:
        return float(atoms.calc.results['energy'])
    return None


def get_ref_stress(atoms):
    """Reference stress of an input frame as a full 3x3, in eV/A^3."""
    if atoms.calc is not None and 'stress' in atoms.calc.results:
        return stress_to_3x3(atoms.calc.results['stress'])
    for key in ('REF_stress', 'stress'):
        if key in atoms.info:
            return stress_to_3x3(atoms.info[key])
    return None


def get_pred_stress(atoms, prefix='MACE_'):
    """Predicted stress of an output frame as a full 3x3, in eV/A^3."""
    key = prefix + 'stress'
    if key in atoms.info:
        return stress_to_3x3(atoms.info[key])
    if atoms.calc is not None and 'stress' in atoms.calc.results:
        return stress_to_3x3(atoms.calc.results['stress'])
    return None


# matscipy's neighbour list is ~40x faster than ASE's and returns the identical
# pair set (verified on 500 MatPES frames and on the edge cases below). It is
# resolved lazily and cached here: None = not yet tried, False = unavailable.
# It raises LinAlgError on a degenerate cell where ASE succeeds, so every call
# falls back to ASE on failure -- correctness never depends on matscipy being
# present or on it handling a particular cell.
_MATSCIPY_NL = None


def _bonded_pairs(atoms, cutoff_scale, radii):
    """
    All pairs (i, j, S, d) with d_ij < cutoff_scale * (radii[i] + radii[j]).

    S is the integer lattice offset of j's image. The trivial i == j / S = 0
    self-pair is excluded; an atom's own SHIFTED images are included, which is
    what lets a cell smaller than one atom be detected.

    matscipy takes a single scalar cutoff, so it is called with the largest
    pair cutoff in the frame and the per-pair rule is applied afterwards. That
    over-searches for light pairs in a frame containing a heavy atom, and is
    still far faster than ASE at every frame size measured.

    Returned pair ORDER differs between the two backends. Nothing downstream
    depends on it: coordination numbers come from a bincount, components from
    connected_components, and the dimensionality is the rank of the cycle
    discrepancies, which is invariant to the spanning tree the BFS happens to
    pick (asserted in the test suite).

    The distance array comes from the backend rather than being recomputed from
    positions and cell. Recomputing looks equivalent but is not on a degenerate
    cell, where the two disagree and a pair can come back with a distance above
    the cutoff it was supposedly selected by.
    """
    global _MATSCIPY_NL
    n = len(atoms)
    if n == 0:
        empty = np.zeros(0, dtype=int)
        return (empty, empty, np.zeros((0, 3), dtype=int),
                np.zeros(0, dtype=float))

    if _MATSCIPY_NL is None:
        try:
            from matscipy.neighbours import neighbour_list as _ms_nl
            _MATSCIPY_NL = _ms_nl
        except ImportError:
            _MATSCIPY_NL = False

    if _MATSCIPY_NL:
        try:
            r_max = cutoff_scale * 2.0 * float(radii.max())
            i, j, S, d = _MATSCIPY_NL('ijSd', atoms, r_max)
            keep = d < cutoff_scale * (radii[i] + radii[j])
            return i[keep], j[keep], S[keep], d[keep]
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            pass          # degenerate cell -> ASE handles it, fall through

    from ase.neighborlist import neighbor_list
    i, j, d, S = neighbor_list('ijdS', atoms, cutoff_scale * radii)
    return i, j, S, d


def min_contact_ratio(atoms, threshold):
    """
    Smallest contact ratio in the cell, or None if nothing is below `threshold`.

    The contact ratio of a pair is its distance divided by the sum of the two
    covalent radii, so 1.0 is "exactly touching" and smaller is compressed.
    The minimum runs over every pair and every periodic image, the i == j /
    zero-offset self-pair excluded but an atom's own *shifted* images included
    -- that is what catches a cell collapsed smaller than a single atom.

    Only pairs below `threshold` are searched for, which keeps the neighbour
    cutoff tiny (~0.8 A) and the call an order of magnitude cheaper than the
    bonded graph. The returned value is still the exact minimum whenever the
    screen fires, because the minimum is by definition below the threshold.
    Verified against a brute-force +-3-image search to 1e-15.
    """
    from ase.data import covalent_radii

    if len(atoms) == 0:
        return None
    rc = covalent_radii[atoms.numbers]
    i, j, _S, d = _bonded_pairs(atoms, threshold, rc)
    if len(i) == 0:
        return None
    ratio = d / (rc[i] + rc[j])
    # Enforce the contract explicitly rather than trusting the backend's own
    # selection: on a degenerate cell a pair can survive with a ratio above the
    # threshold, and returning it would label an uncompressed frame as
    # overcompressed. For a well-formed cell this filter removes nothing.
    ratio = ratio[ratio < threshold]
    if ratio.size == 0:
        return None
    return float(ratio.min())


def structure_descriptors(atoms, bond_scale=NON_BULK_BOND_SCALE,
                          need_graph=True):
    """
    Geometric descriptors used to decide whether a cell is bulk-like.

    packing        sum of covalent-sphere volumes / cell volume. See the long
                   note by NON_BULK_PACKING_THRESHOLD: this is a density index,
                   not a packing fraction, and it CANNOT classify bulk vs
                   non-bulk on its own.
    dim            periodic dimensionality, 0-3, of the bonded network:
                   3 = ordinary bulk, 2 = slab or layer, 1 = chain,
                   0 = isolated cluster or molecule.
    n_fragments    connected components under the bond cutoff.
    min_cn         smallest coordination number in the cell.

    How `dim` is computed, and why not the obvious way
    -------------------------------------------------
    The tempting shortcut is to take the rank of the lattice offsets of all
    bonds that leave the home cell. That is WRONG, and wrong in the dangerous
    direction -- it over-estimates. A four-atom molecule whose bonds happen to
    cross the x, y and z boundaries in a chain has three crossing offsets of
    rank 3, and the shortcut calls it 3D bulk when it is an isolated molecule.

    The correct quantity (Larsen et al., Phys. Rev. Mater. 3, 034003 (2019)) is
    the rank of the CYCLE discrepancies. Assign every atom a lattice offset
    p_i by breadth-first search over the bond graph, following each tree edge
    (i -> j, offset S) with p_j = p_i + S. Then every edge has a discrepancy

            d = p_i + S - p_j

    which is zero on tree edges by construction, and on any other edge is a
    lattice vector under which the fragment maps onto itself. The rank of those
    discrepancies, per connected fragment, is the dimensionality. Validated
    against pymatgen's get_dimensionality_larsen (8/8 agreement) and on
    hand-built cases including the counterexample above.

    Degenerate cells
    ----------------
    A frame with no cell, a zero cell or a planar cell has no defined volume;
    ASE raises on get_volume(). Such a frame cannot be a 3D bulk crystal, so
    packing is reported as 0.0 (which flags it non-bulk under any positive
    threshold) rather than letting the exception kill the whole run.

    With need_graph=False only `packing` is computed and the graph fields come
    back as None; that path costs microseconds.

    ASE's neighbour list is used rather than a hand-rolled image loop: it tiles
    the cell as far as the cutoff actually requires. A fixed +-1 image window is
    wrong for cells with a short or strongly skewed lattice vector, where bonds
    reach offsets of several cells.
    """
    from ase.data import covalent_radii

    n = len(atoms)
    rc = covalent_radii[atoms.numbers]
    try:
        volume = abs(atoms.get_volume())
    except (ValueError, np.linalg.LinAlgError):
        volume = 0.0                      # no cell / degenerate cell
    if volume > MIN_CELL_VOLUME:
        packing = float(((4.0 / 3.0) * np.pi * rc ** 3).sum() / volume)
        vol_per_atom = volume / n if n else float('nan')
    else:
        # Not a periodic bulk cell: flag it rather than raising or, worse,
        # returning a value that reads as "dense".
        packing = 0.0
        vol_per_atom = float('nan')
    out = {'packing': packing, 'volume': volume,
           'volume_per_atom': vol_per_atom,
           'dim': None, 'n_fragments': None, 'min_cn': None}
    if not need_graph:
        return out

    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg

    if n == 0:
        out.update(dim=0, n_fragments=0, min_cn=0)
        return out
    try:
        i, j, offsets, _d = _bonded_pairs(atoms, bond_scale, rc)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        # no backend could build a neighbour list; treat as isolated atoms
        out.update(dim=0, n_fragments=n, min_cn=0)
        return out
    cn = np.bincount(i, minlength=n)
    out['min_cn'] = int(cn.min())
    if len(i) == 0:
        out['n_fragments'] = n          # every atom isolated
        out['dim'] = 0
        return out
    adj = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    n_comp, labels = csg.connected_components(adj, directed=False)
    out['n_fragments'] = int(n_comp)

    # --- lattice offset per atom, by BFS over the bond graph ---------------
    nbrs = [[] for _ in range(n)]
    for a, b, sh in zip(i, j, offsets):
        nbrs[a].append((b, sh))
    p = np.zeros((n, 3), dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, sh in nbrs[u]:
                if not seen[v]:
                    seen[v] = True
                    p[v] = p[u] + sh
                    queue.append(v)

    # --- rank of the cycle discrepancies, per fragment ---------------------
    disc = p[i] + offsets - p[j]
    nonzero = np.any(disc != 0, axis=1)
    dim = 0
    if nonzero.any():
        comp_of_bond = labels[i][nonzero]
        dd = disc[nonzero]
        for c in np.unique(comp_of_bond):
            dim = max(dim, int(np.linalg.matrix_rank(dd[comp_of_bond == c])))
            if dim == 3:
                break
    out['dim'] = dim
    return out


# name -> (label, needs the bonded graph, predicate on the descriptor dict)
NON_BULK_CRITERIA = {
    'packing': (
        f'packing fraction < {{thr:g}}', False,
        lambda d, thr: d['packing'] < thr),
    'dim': (
        'periodic dimensionality < 3', True,
        lambda d, thr: d['dim'] < 3),
    'fragmented': (
        'more than one connected fragment', True,
        lambda d, thr: d['n_fragments'] > 1),
    'dim_or_frag': (
        'dimensionality < 3 or fragmented', True,
        lambda d, thr: d['dim'] < 3 or d['n_fragments'] > 1),
    'min_cn': (
        'an atom with coordination number <= 2', True,
        lambda d, thr: d['min_cn'] <= 2),
    'any': (
        'low packing, or dim < 3, or fragmented, or an atom with CN <= 2', True,
        lambda d, thr: (d['packing'] < thr or d['dim'] < 3
                        or d['n_fragments'] > 1 or d['min_cn'] <= 2)),
}


def resolve_non_bulk_detector(name):
    """(label, needs_graph, predicate) for a detector name; None -> disabled."""
    if name is None:
        return None
    if name not in NON_BULK_CRITERIA:
        raise ValueError(
            f'NON_BULK_DETECTOR must be one of {sorted(NON_BULK_CRITERIA)} or '
            f'None, got {name!r}')
    label, needs_graph, fn = NON_BULK_CRITERIA[name]
    return label, needs_graph, fn


def has_usable_cell(atoms):
    """True if the frame has a finite, non-degenerate cell (stress is defined)."""
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    if not np.isfinite(cell).all():
        return False
    return abs(float(np.linalg.det(cell))) > MIN_CELL_VOLUME


# ---------------------------------------------------------------------------
# Frame indexing / subsampling
# ---------------------------------------------------------------------------
def build_frame_offsets(path):
    """
    Byte offset of the start of every frame in an extxyz file.

    One sequential pass, no parsing (~1 us/frame), which gives O(1) random
    access to any frame afterwards. Blank lines between frames are tolerated.
    """
    offsets = []
    with open(path, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue                      # blank line between frames
            n_atoms = int(stripped)
            offsets.append(pos)
            f.readline()                      # comment line
            for _ in range(n_atoms):
                f.readline()                  # atom lines
    return offsets


def select_frame_indices(n_frames, fraction, seed):
    """
    Sorted indices of the frames to keep.

    Deterministic in (seed, fraction, n_frames) alone -- deliberately NOT a
    function of the model, so the identical subset is drawn for every model you
    compare on the same reference file.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f'FRACTION must be in (0, 1], got {fraction}')
    if fraction == 1.0:
        return np.arange(n_frames)
    n_keep = int(round(fraction * n_frames))
    n_keep = max(1, min(n_frames, n_keep))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_frames, size=n_keep, replace=False))


def write_frame_subset(src_path, offsets, indices, dst_path):
    """
    Copy the selected frames to dst_path as raw bytes.

    Byte-exact: no ASE round-trip, so every info key, column and number of
    significant digits in the reference file survives untouched.
    """
    n_frames = len(offsets)
    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
        for i in indices:
            start = offsets[i]
            end = offsets[i + 1] if i + 1 < n_frames else None
            src.seek(start)
            chunk = src.read(end - start) if end is not None else src.read()
            if not chunk.endswith(b'\n'):
                chunk += b'\n'
            dst.write(chunk)
    return dst_path


# ---------------------------------------------------------------------------
# Error accumulation and reporting
# ---------------------------------------------------------------------------
class _Accumulator:
    """Sum of |e| and e^2 over a stream of errors, plus a count."""

    __slots__ = ('sum_abs', 'sum_sq', 'n')

    def __init__(self):
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.n = 0

    def add(self, errors):
        e = np.asarray(errors, dtype=np.float64)
        self.sum_abs += float(np.abs(e).sum())
        self.sum_sq += float(np.square(e).sum())
        self.n += int(e.size)

    @property
    def mae(self):
        return self.sum_abs / self.n if self.n else float('nan')

    @property
    def rmse(self):
        return (self.sum_sq / self.n) ** 0.5 if self.n else float('nan')


def _print_pair(label, acc, scale, unit_small, unit_base, indent=2):
    """One MAE/RMSE pair, printed in both the milli- and the base unit."""
    pad = ' ' * indent
    print(f'{pad}{label}')
    print(f'{pad}  MAE  = {acc.mae * scale:14.4f} {unit_small}'
          f'   ({acc.mae:.6f} {unit_base})')
    print(f'{pad}  RMSE = {acc.rmse * scale:14.4f} {unit_small}'
          f'   ({acc.rmse:.6f} {unit_base})')


class _MetricSet:
    """
    Every accumulator for one selection of the data.

    One of these holds the whole report for a single (variant, cutoff) choice,
    so the uncut report and each cutoff block are computed by exactly the same
    code and cannot drift apart.
    """

    def __init__(self):
        self.e_struct = _Accumulator()   # |dE| per structure, eV
        self.e_atom = _Accumulator()     # |dE|/N per structure, eV/atom
        self.f_norm = _Accumulator()     # ||dF_i|| per atom, eV/A
        self.f_comp = _Accumulator()     # dF per component, eV/A
        self.s_norm = _Accumulator()     # ||dS||_F per structure, eV/A^3
        self.s_comp9 = _Accumulator()    # dS over all 9 elements, eV/A^3
        self.s_comp6 = _Accumulator()    # dS over Voigt-6, eV/A^3
        self.n_frames = 0                # structures fed into this selection
        self.n_atoms = 0                 # atoms kept in the force statistics

    def feed(self, n_at, d_e, f_diff, f_norms, s_diff, atom_mask=None):
        """
        Add one frame. Any of d_e / f_diff / s_diff may be None (that quantity
        is then simply not counted for this frame). atom_mask, if given, is a
        boolean array selecting which atoms enter the force statistics.
        """
        self.n_frames += 1
        if d_e is not None:
            self.e_struct.add(d_e)
            self.e_atom.add(d_e / n_at)
        if f_diff is not None:
            if atom_mask is None:
                sel_diff, sel_norms = f_diff, f_norms
            else:
                sel_diff, sel_norms = f_diff[atom_mask], f_norms[atom_mask]
            if sel_norms.size:
                self.f_norm.add(sel_norms)
                self.f_comp.add(sel_diff)
                self.n_atoms += int(sel_norms.size)
        if s_diff is not None:
            self.s_norm.add(np.linalg.norm(s_diff))
            self.s_comp9.add(s_diff)
            self.s_comp6.add(np.array(
                [s_diff[0, 0], s_diff[1, 1], s_diff[2, 2],
                 s_diff[1, 2], s_diff[0, 2], s_diff[0, 1]]))

    def as_dict(self, prefix=''):
        return {
            prefix + 'mae_energy': self.e_struct.mae,
            prefix + 'rmse_energy': self.e_struct.rmse,
            prefix + 'mae_energy_per_atom': self.e_atom.mae,
            prefix + 'rmse_energy_per_atom': self.e_atom.rmse,
            prefix + 'mae_forces_norm': self.f_norm.mae,
            prefix + 'rmse_forces_norm': self.f_norm.rmse,
            prefix + 'mae_forces_component': self.f_comp.mae,
            prefix + 'rmse_forces_component': self.f_comp.rmse,
            prefix + 'mae_stress_norm': self.s_norm.mae,
            prefix + 'rmse_stress_norm': self.s_norm.rmse,
            prefix + 'mae_stress_component': self.s_comp9.mae,
            prefix + 'rmse_stress_component': self.s_comp9.rmse,
            prefix + 'mae_stress_voigt6': self.s_comp6.mae,
            prefix + 'rmse_stress_voigt6': self.s_comp6.rmse,
            prefix + 'n_structures_energy': self.e_struct.n,
            prefix + 'n_structures_stress': self.s_norm.n,
            prefix + 'n_atoms': self.n_atoms,
        }


def _print_metric_set(ms, indent=0, es_note=''):
    """Print the ENERGY / FORCES / STRESS sections of one _MetricSet."""
    pad = ' ' * indent
    if ms.e_struct.n:
        print(f'{pad}ENERGY  ({ms.e_struct.n} structures){es_note}')
        _print_pair('Per structure (e = |E_pred - E_ref|)',
                    ms.e_struct, 1000.0, 'meV     ', 'eV', indent + 2)
        _print_pair('Per atom (MACE log convention, RMSE_E_per_atom)',
                    ms.e_atom, 1000.0, 'meV/atom', 'eV/atom', indent + 2)
    else:
        print(f'{pad}ENERGY  no structures left in this selection')

    if ms.f_norm.n:
        print(f'{pad}FORCES  ({ms.f_norm.n} atoms, {ms.f_comp.n} components)')
        _print_pair('Norm-based (per atom, e_i = ||F_pred,i - F_ref,i||)',
                    ms.f_norm, 1000.0, 'meV/A   ', 'eV/A', indent + 2)
        _print_pair('Component-wise (per force component, MACE log '
                    'convention, RMSE_F)',
                    ms.f_comp, 1000.0, 'meV/A   ', 'eV/A', indent + 2)
    else:
        print(f'{pad}FORCES  no atoms left in this selection')

    if ms.s_norm.n:
        print(f'{pad}STRESS  ({ms.s_norm.n} structures, {ms.s_comp9.n} '
              f'components){es_note}')
        _print_pair('Norm-based (per structure, e = ||S_pred - S_ref||_F)',
                    ms.s_norm, 1000.0, 'meV/A^3 ', 'eV/A^3', indent + 2)
        _print_pair('Component-wise, full 3x3 (9 per structure, MACE log '
                    'convention, RMSE_stress)',
                    ms.s_comp9, 1000.0, 'meV/A^3 ', 'eV/A^3', indent + 2)
        _print_pair('Component-wise, Voigt-6 (6 per structure, off-diagonals '
                    'counted once)',
                    ms.s_comp6, 1000.0, 'meV/A^3 ', 'eV/A^3', indent + 2)
    else:
        print(f'{pad}STRESS  no structures left in this selection')


def _clean_cutoffs(cutoffs):
    """Validate, de-duplicate and sort the requested force-error cutoffs."""
    if not cutoffs:
        return []
    out = []
    for c in cutoffs:
        c = float(c)
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError(
                f'FORCE_ERROR_CUTOFFS must contain finite positive values in '
                f'eV/A, got {c}')
        out.append(c)
    return sorted(set(out))


def report_efs_errors(ref_path, pred_path, info_prefix='MACE_',
                      sample_info=None, force_error_cutoffs=None,
                      cutoff_on=None, non_bulk_detector='__default__',
                      non_bulk_packing_threshold=None,
                      contact_ratio_cutoff='__default__',
                      collect_per_structure=None, paper_dir=None,
                      ceiling_path=None, save_per_atom=None,
                      save_per_atom_forces=None,
                      model_label=None):
    """
    Stream ref_path and pred_path in lockstep and print energy/force/stress
    MAE and RMSE in both the norm-based and the component-wise convention.

    force_error_cutoffs: thresholds in eV/A. For each one the same metrics are
    printed again with the atoms above the threshold excluded, in the two
    variants described in the module docstring. None -> use the
    FORCE_ERROR_CUTOFFS setting.

    cutoff_on: which per-atom statistic the threshold applies to,
    'max_component' (component-wise) or 'norm'. None -> use the
    FORCE_ERROR_CUTOFF_ON setting.
    """
    if force_error_cutoffs is None:
        force_error_cutoffs = FORCE_ERROR_CUTOFFS
    cutoffs = _clean_cutoffs(force_error_cutoffs)
    if cutoff_on is None:
        cutoff_on = FORCE_ERROR_CUTOFF_ON
    if cutoff_on not in CUTOFF_CRITERIA:
        raise ValueError(
            f'cutoff_on must be one of {sorted(CUTOFF_CRITERIA)}, got '
            f'{cutoff_on!r}')
    crit_label, crit_fn = CUTOFF_CRITERIA[cutoff_on]

    if non_bulk_detector == '__default__':
        non_bulk_detector = NON_BULK_DETECTOR
    nb_spec = resolve_non_bulk_detector(non_bulk_detector)
    nb_thr = (NON_BULK_PACKING_THRESHOLD if non_bulk_packing_threshold is None
              else float(non_bulk_packing_threshold))
    if contact_ratio_cutoff == '__default__':
        contact_ratio_cutoff = CONTACT_RATIO_CUTOFF
    if contact_ratio_cutoff is not None:
        contact_ratio_cutoff = float(contact_ratio_cutoff)
        if not 0.0 < contact_ratio_cutoff <= 2.0:
            raise ValueError(
                f'CONTACT_RATIO_CUTOFF must be in (0, 2], got '
                f'{contact_ratio_cutoff}')

    everything = _MetricSet()
    # non-bulk split: same metrics over the two halves of the test set
    bulk_set, nonbulk_set, compressed_set = (_MetricSet(), _MetricSet(),
                                             _MetricSet())
    n_nonbulk = 0
    n_compressed = 0
    worst_contacts = []          # (ratio, frame index) for the report

    # Domains the cutoff blocks are broken down over. 'all' is always there;
    # the others exist only if the corresponding detector is active, which is
    # exactly the rule the uncut DOMAIN SPLIT section follows.
    dom_keys = ['all']
    if nb_spec is not None or contact_ratio_cutoff is not None:
        dom_keys.append('bulk')
        if nb_spec is not None:
            dom_keys.append('non_bulk')
        if contact_ratio_cutoff is not None:
            dom_keys.append('overcompressed')

    # per cutoff and per domain: [a] whole structures dropped,
    #                            [b] offending atoms dropped
    dropped = {c: {k: _MetricSet() for k in dom_keys} for c in cutoffs}
    kept = {c: {k: _MetricSet() for k in dom_keys} for c in cutoffs}
    # how much each cutoff removes, per domain
    n_atoms_above = {c: {k: 0 for k in dom_keys} for c in cutoffs}
    n_frames_affected = {c: {k: 0 for k in dom_keys} for c in cutoffs}
    sq_above = {c: {k: 0.0 for k in dom_keys} for c in cutoffs}
    n_unclassifiable = {c: {k: 0 for k in dom_keys} for c in cutoffs}
    # Denominator of the "share of squared force error" line, per domain. Only
    # 'all' and the frame's own domain are incremented, so these sum correctly.
    sq_total_dom = {k: 0.0 for k in dom_keys}

    if collect_per_structure is None:
        collect_per_structure = COLLECT_PER_STRUCTURE
    if save_per_atom is None:
        save_per_atom = SAVE_PER_ATOM
    collect = (_PerStructure(save_per_atom=save_per_atom,
                             save_per_atom_forces=save_per_atom_forces)
               if collect_per_structure else None)

    n_frames = 0
    n_missing_e = n_missing_f = n_missing_s = 0
    n_nofinite_e = n_nofinite_f = n_nofinite_s = 0
    n_no_cell = 0

    ref_iter = ase.io.iread(ref_path, index=':', format='extxyz')
    pred_iter = ase.io.iread(pred_path, index=':', format='extxyz')

    # zip_longest, not zip: plain zip() stops at the shorter file *after*
    # having already pulled and discarded the extra frame from the longer one,
    # so a length mismatch cannot be detected afterwards. The None fill value
    # makes it explicit at the frame where one file runs out.
    for frame_idx, (ref_atoms, pred_atoms) in enumerate(
            zip_longest(ref_iter, pred_iter)):
        if ref_atoms is None or pred_atoms is None:
            longer = pred_path if ref_atoms is None else ref_path
            raise ValueError(
                f'Frame-count mismatch at frame {frame_idx}: {longer} contains '
                f'more structures than the other file. {ref_path} and '
                f'{pred_path} are not aligned.')
        if len(ref_atoms) != len(pred_atoms):
            raise ValueError(
                f'Frame {frame_idx}: atom-count mismatch between reference '
                f'({len(ref_atoms)}) and prediction ({len(pred_atoms)}) - the '
                f'two files are not aligned.')
        if ref_atoms.get_chemical_symbols() != pred_atoms.get_chemical_symbols():
            raise ValueError(
                f'Frame {frame_idx}: species mismatch between reference and '
                f'prediction - the two files are not aligned.')

        n_frames += 1
        n_at = len(ref_atoms)

        # --- energy --------------------------------------------------------
        e_ref = get_ref_energy(ref_atoms)
        e_pred = get_pred_energy(pred_atoms, prefix=info_prefix)
        d_e = None
        if e_ref is None or e_pred is None:
            n_missing_e += 1
        else:
            d_e = e_pred - e_ref
            if not np.isfinite(d_e):
                n_nofinite_e += 1

        # --- forces --------------------------------------------------------
        f_ref = get_ref_forces(ref_atoms)
        f_pred = get_pred_forces(pred_atoms, prefix=info_prefix)
        f_diff = f_norms = f_crit = None
        frame_sq = 0.0               # sum_i ||dF_i||^2 for this frame
        if f_ref is None or f_pred is None:
            n_missing_f += 1
        else:
            f_diff = f_pred - f_ref                    # (N, 3) component-wise
            if not np.isfinite(f_diff).all():
                n_nofinite_f += 1
            f_norms = np.linalg.norm(f_diff, axis=1)   # (N,) per-atom norm
            # The statistic the cutoff is tested against; may differ from the
            # norm used by the norm-based metric.
            f_crit = crit_fn(f_diff, f_norms)
            frame_sq = float(np.square(f_norms).sum())

        # --- stress --------------------------------------------------------
        s_ref = get_ref_stress(ref_atoms)
        s_pred = get_pred_stress(pred_atoms, prefix=info_prefix)
        s_diff = None
        if s_ref is None or s_pred is None:
            n_missing_s += 1
        elif not has_usable_cell(ref_atoms):
            # MACE clamps stress to 0 for a degenerate cell; that is not a
            # prediction, so it must not enter the statistics.
            n_no_cell += 1
        else:
            s_diff = s_pred - s_ref                    # (3, 3)
            if not np.isfinite(s_diff).all():
                n_nofinite_s += 1

        # --- feed the uncut report -----------------------------------------
        everything.feed(n_at, d_e, f_diff, f_norms, s_diff)

        # --- domain classification ------------------------------------------
        # Compressed first: an overcompressed cell is a reference-data problem,
        # and it is usually DENSE, so it would otherwise be counted as bulk.
        # domain is the key the cutoff blocks below are grouped by; None when
        # no split is active, in which case only 'all' is reported.
        domain = None
        desc = None
        if nb_spec is not None or contact_ratio_cutoff is not None:
            contact = (min_contact_ratio(ref_atoms, contact_ratio_cutoff)
                       if contact_ratio_cutoff is not None else None)
            if contact is not None:
                domain = 'overcompressed'
                n_compressed += 1
                worst_contacts.append((contact, frame_idx))
                compressed_set.feed(n_at, d_e, f_diff, f_norms, s_diff)
            elif nb_spec is None:
                domain = 'bulk'
                bulk_set.feed(n_at, d_e, f_diff, f_norms, s_diff)
            else:
                _, needs_graph, nb_fn = nb_spec
                desc = structure_descriptors(ref_atoms,
                                             need_graph=needs_graph)
                if bool(nb_fn(desc, nb_thr)):
                    domain = 'non_bulk'
                    n_nonbulk += 1
                    nonbulk_set.feed(n_at, d_e, f_diff, f_norms, s_diff)
                else:
                    domain = 'bulk'
                    bulk_set.feed(n_at, d_e, f_diff, f_norms, s_diff)

        # --- feed each cutoff ----------------------------------------------
        # The frame belongs to 'all' and, when a split is active, to exactly one
        # geometric domain. Both get identical treatment, so every cut metric
        # exists per domain as well as pooled.
        frame_doms = ('all',) if domain is None else ('all', domain)
        for k in frame_doms:
            sq_total_dom[k] += frame_sq
        for c in cutoffs:
            if f_norms is None:
                # Without forces the frame cannot be classified as clean or
                # not, so it cannot enter variant [a]. It still contributes its
                # energy and stress to variant [b], which keeps every structure.
                for k in frame_doms:
                    n_unclassifiable[c][k] += 1
                    kept[c][k].feed(n_at, d_e, None, None, s_diff)
                continue
            above = f_crit > c
            n_bad = int(above.sum())
            sq_bad = float(np.square(f_norms[above]).sum()) if n_bad else 0.0
            for k in frame_doms:
                if n_bad:
                    n_atoms_above[c][k] += n_bad
                    n_frames_affected[c][k] += 1
                    sq_above[c][k] += sq_bad
                # [a] keep the structure only if every atom is at or below c
                if not n_bad:
                    dropped[c][k].feed(n_at, d_e, f_diff, f_norms, s_diff)
                # [b] drop only the offending atoms; keep the structure for E
                # and S
                kept[c][k].feed(n_at, d_e, f_diff, f_norms, s_diff,
                                atom_mask=~above)

        # --- v2: one row per structure -------------------------------------
        if collect is not None:
            if desc is None or desc.get('dim') is None:
                desc = structure_descriptors(ref_atoms, need_graph=True)
            collect.add(frame_idx, ref_atoms, n_at,
                        'all' if domain is None else domain,
                        desc, d_e, f_diff, f_norms, s_diff, f_ref, s_ref)

    if not (everything.e_struct.n or everything.f_norm.n
            or everything.s_norm.n):
        raise ValueError(
            'No frame carried a reference/prediction pair for any of energy, '
            'forces or stress - nothing to report. Check that the reference '
            'file has energy/forces/stress and that --info_prefix matches the '
            'eval output.')

    # ---- report -----------------------------------------------------------
    print('')
    print('=' * 72)
    print('Energy / force / stress error report')
    print(f'  reference : {ref_path}')
    print(f'  prediction: {pred_path}')
    print(f'  {n_frames} structures, {everything.n_atoms} atoms '
          f'({everything.f_comp.n} force components, '
          f'{everything.s_comp9.n} stress components)')
    if sample_info is not None:
        print(f'  RANDOM SUBSET: fraction={sample_info["fraction"]:g}, '
              f'seed={sample_info["seed"]}, '
              f'{sample_info["n_selected"]} of {sample_info["n_total"]} frames')
        print(f'  (same seed+fraction+reference file -> same frames for any '
              f'model; RMSE is noisy under subsampling, see module docstring)')
    for what, n_missing in (('energy', n_missing_e), ('forces', n_missing_f),
                            ('stress', n_missing_s)):
        if n_missing:
            print(f'  {n_missing} structures without a reference/predicted '
                  f'{what} pair (excluded from the {what} statistics)')
    if n_no_cell:
        print(f'  {n_no_cell} structures with no usable cell (excluded from '
              f'the stress statistics only)')
    for what, n_bad in (('energy', n_nofinite_e), ('forces', n_nofinite_f),
                        ('stress', n_nofinite_s)):
        if n_bad:
            print(f'  WARNING: {n_bad} structures carry non-finite {what} '
                  f'errors; they are KEPT, so the {what} metrics below are '
                  f'nan. This is a diverging model, not a reporting bug.')

    print('=' * 72)
    print('ALL DATA -- no force-error cutoff')
    print('-' * 72)
    if everything.s_norm.n == 0:
        _print_metric_set(everything)
        print('        Stress: eval_configs only writes MACE_stress when it '
              'is called with')
        print('        --compute_stress. If you used --skip_eval on an output '
              'file from')
        print('        the forces-only script, that file has no stress column '
              '- re-run')
        print('        the evaluation. If the reference file stores a virial '
              'instead of')
        print('        a stress, convert it to a stress first (divide by the '
              'cell volume).')
    else:
        _print_metric_set(everything)

    if nb_spec is not None or contact_ratio_cutoff is not None:
        n_bulk = n_frames - n_nonbulk - n_compressed
        print('=' * 72)
        print('DOMAIN SPLIT')
        if contact_ratio_cutoff is not None:
            print(f'  overcompressed: min contact ratio < '
                  f'{contact_ratio_cutoff:g}')
            print(f'                  (contact ratio = pair distance / sum of '
                  f'covalent radii,')
            print(f'                   minimised over all pairs and all '
                  f'periodic images)')
        if nb_spec is not None:
            nb_label, _, _ = nb_spec
            print(f'  non-bulk      : {nb_label.format(thr=nb_thr)}   '
                  f'[detector: {non_bulk_detector}]')
        print('-' * 72)
        rows = [('BULK', n_bulk, bulk_set)]
        if nb_spec is not None:
            rows.append(('NON-BULK', n_nonbulk, nonbulk_set))
        if contact_ratio_cutoff is not None:
            rows.append(('OVERCOMPRESSED', n_compressed, compressed_set))
        for name, cnt, _ms in rows:
            print(f'  {name:15s} {cnt:7d} structures '
                  f'({100.0 * cnt / n_frames:6.2f}%)')
        print('  Nothing is discarded: every structure is in exactly one of the')
        print('  domains below, so no exclusion rule is applied to the numbers.')
        if worst_contacts:
            worst_contacts.sort()
            shown = ', '.join(f'{r:.3f}' for r, _ in worst_contacts[:8])
            print(f'  smallest contact ratios seen: {shown}'
                  + (' ...' if len(worst_contacts) > 8 else ''))
        for name, cnt, ms in rows:
            print('-' * 72)
            print(f'  {name} domain')
            if cnt == 0:
                print('      (empty)')
            else:
                _print_metric_set(ms, indent=6)

    # Labels and totals for the per-cutoff domain breakdown, in the same order
    # as the uncut DOMAIN SPLIT section above. Each domain is measured against
    # its own structure count, atom count and summed squared force error.
    dom_totals = {
        'all': ('ALL DATA', n_frames, everything.n_atoms),
        'bulk': ('BULK', n_frames - n_nonbulk - n_compressed,
                 bulk_set.n_atoms),
        'non_bulk': ('NON-BULK', n_nonbulk, nonbulk_set.n_atoms),
        'overcompressed': ('OVERCOMPRESSED', n_compressed,
                           compressed_set.n_atoms),
    }

    for c in cutoffs:
        print('=' * 72)
        print(f'FORCE-ERROR CUTOFF {c:g} eV/A   on {cutoff_on}')
        print(f'  criterion: atom excluded when {crit_label} > {c:g} eV/A')
        if len(dom_keys) > 1:
            print('  Reported for ALL DATA and then for every domain of the '
                  'DOMAIN SPLIT above.')
            print('  Counts and percentages inside a domain are taken over '
                  'that domain only.')
        for k in dom_keys:
            dom_label, n_dom_frames, n_dom_atoms = dom_totals[k]
            print('-' * 72)
            print(f'  {dom_label}  ({n_dom_frames} structures, '
                  f'{n_dom_atoms} atoms)')
            if n_dom_frames == 0:
                print('      (empty)')
                continue
            pct_atoms = (100.0 * n_atoms_above[c][k] / n_dom_atoms
                         if n_dom_atoms else float('nan'))
            pct_frames = 100.0 * n_frames_affected[c][k] / n_dom_frames
            dom_sq = sq_total_dom[k]
            pct_sq = (100.0 * sq_above[c][k] / dom_sq
                      if dom_sq > 0.0 else float('nan'))
            whose = ('the total squared force error' if k == 'all'
                     else 'this domain\'s squared force error')
            print(f'      atoms above cutoff      : {n_atoms_above[c][k]} of '
                  f'{n_dom_atoms} ({pct_atoms:.4g}%), carrying '
                  f'{pct_sq:.4g}% of {whose}')
            print(f'      structures with >=1 such atom: '
                  f'{n_frames_affected[c][k]} of {n_dom_frames} '
                  f'({pct_frames:.4g}%)')
            if n_unclassifiable[c][k]:
                print(f'      {n_unclassifiable[c][k]} structures have no '
                      f'forces and so cannot be classified; they are absent '
                      f'from [a] and present in [b]')
            print(f'      [a] whole structures dropped -- '
                  f'{dropped[c][k].n_frames} of {n_dom_frames} structures '
                  f'survive')
            print(f'          (energy, forces and stress all over the same '
                  f'surviving structures)')
            _print_metric_set(dropped[c][k], indent=10)
            print(f'      [b] offending atoms dropped, all '
                  f'{kept[c][k].n_frames} structures kept')
            _print_metric_set(
                kept[c][k], indent=10,
                es_note=f'   [= {dom_label} uncut, atom cut cannot change it]')

    print('=' * 72)

    paper = None
    if collect is not None:
        rows = collect.finish()
        paper = _paper_report(rows, paper_dir=paper_dir,
                              ceiling_path=ceiling_path,
                              model_label=model_label,
                              ref_path=ref_path, pred_path=pred_path,
                              non_bulk_detector=non_bulk_detector,
                              contact_ratio_cutoff=contact_ratio_cutoff,
                              per_atom=collect.per_atom_arrays(),
                              peratom_hists=collect.hists(),
                              per_atom_forces=collect.per_atom_force_arrays())

    out = everything.as_dict()
    out['n_structures'] = n_frames
    out['n_atoms'] = everything.n_atoms
    out['cutoff_on'] = cutoff_on
    if nb_spec is not None or contact_ratio_cutoff is not None:
        out['domains'] = {
            'detector': non_bulk_detector if nb_spec is not None else None,
            'packing_threshold': nb_thr if nb_spec is not None else None,
            'contact_ratio_cutoff': contact_ratio_cutoff,
            'n_bulk': n_frames - n_nonbulk - n_compressed,
            'n_non_bulk': n_nonbulk,
            'n_overcompressed': n_compressed,
            'bulk': bulk_set.as_dict(),
            'non_bulk': nonbulk_set.as_dict(),
            'overcompressed': compressed_set.as_dict(),
            'min_contact_ratios': sorted(r for r, _ in worst_contacts),
        }
    def _cutoff_entry(c, k):
        """The cut metrics for one cutoff on one domain."""
        dom_sq = sq_total_dom[k]
        return {
            'n_structures': dom_totals[k][1],
            'n_atoms': dom_totals[k][2],
            'n_atoms_above': n_atoms_above[c][k],
            'n_structures_affected': n_frames_affected[c][k],
            'fraction_squared_force_error_above': (
                sq_above[c][k] / dom_sq if dom_sq > 0.0 else float('nan')),
            'structures_dropped': dropped[c][k].as_dict(),
            'structures_kept': kept[c][k].as_dict(),
            'n_structures_surviving': dropped[c][k].n_frames,
        }

    # The top-level keys stay on ALL DATA, so anything already reading this
    out['cutoffs'] = {
        c: dict(_cutoff_entry(c, 'all'),
                domains={k: _cutoff_entry(c, k) for k in dom_keys})
        for c in cutoffs
    }
    if paper is not None:
        out['paper'] = paper
    return out


# ===========================================================================
# v2: per-structure capture, the paper tables, and multi-model aggregation
# ===========================================================================
# WHY THIS EXISTS
# ---------------
# The v1 report is a pooled MAE/RMSE over every atom of the test set. On MatPES
# that statistic is dominated by a handful of structures: measured over the six
# 400k-subset models, 1-3 structures out of 323,096 carry 50% of the total
# squared force error and 6-37 carry 99%. Four models that differ only in the
# random seed of their training subset then span a factor of 88 in pooled force
# MAE (405 - 35,577 meV/A) and a factor of 307 in pooled RMSE. No model
# comparison survives that, in either direction.
#
# The fix is not to discard the outliers -- that is what invites the charge of
# cherry-picking, and a robust average can genuinely hide a model that fails on
# a physically relevant minority (Morrow, Gardner & Deringer, arXiv:2211.12484).
# The fix is to report the whole distribution and to state the noise floor. So
# this section adds, per domain:
#
#   ROBUST ACCURACY      median / quantiles / trimmed mean / geometric mean of
#                        the PER-STRUCTURE error, next to the pooled mean.
#   STABILITY            counts of structures above fixed error thresholds,
#                        reported separately from accuracy so a model that is
#                        typically accurate but occasionally divergent cannot
#                        hide behind either number alone.
#   ERROR CONCENTRATION  how many structures carry 50/90/99/99.9% of the total
#                        squared error. This is the diagnostic that tells the
#                        reader why the pooled RMSE cannot be quoted.
#   CDF                  fraction of structures below a grid of thresholds,
#                        also written as CSV for the paper figure.
#   REFERENCE-FORCE      the same metrics split by max |F_ref|. Model-
#   SPLIT                independent by construction: computed from the DFT
#                        reference alone, so it cannot favour any model.
#   CEILING-RELATIVE     with --ceiling, the paired difference and ratio
#                        against a CEILING MODEL trained on the full dataset
#                        ceiling's own error. Pairing per structure cancels the
#                        intrinsic difficulty of the structure and is what
#                        makes a 3 meV/A effect resolvable at all.
#
# And with --aggregate it reads the per-structure tables of several models and
# emits the combined paper tables, including the random-seed mean +- sd and
# each model's offset from it in units of that sd.
#
# WHAT NOT TO DO WITH THIS OUTPUT
# -------------------------------
# Do not define a domain from any model's errors, the ceiling model's included.
# inside the selection (on MatPES, thresholding MACE-OMAT-0's per-structure
# error at 1 eV/A improves its own median from 101.9 to 97.9 meV/A purely by
# fails. The splits offered here are all computed from the reference data or
# the training data instead. The ceiling-conditional profile does bin on the
# ceiling's error, which is why it prints every bin and why its header states
# that only the candidate models may be compared within a bin -- none of them
# was used to define the bins, so their relative ranking is unaffected, while
# the ceiling's own column inside a bin is not a fair estimate of its skill.


def _default_label(model_path):
    """A short, filesystem-safe name for a model, derived from its path."""
    base = os.path.basename(str(model_path)).replace('.model', '')
    base = base.replace('_compiled', '')
    return base[:64] if base else 'model'


def min_contact_ratio_full(atoms, start_scale=1.0, max_scale=4.0):
    """
    The exact minimum contact ratio of a cell, with no threshold.

    min over all pairs (i, j) and all lattice offsets o, excluding only the
    i == j / o = 0 self-pair, of |r_i - (r_j + o.A)| / (R_i + R_j).

    `min_contact_ratio` above answers only "is anything below `threshold`",
    which is all the domain screen needs. For the saved table we want the value
    itself, so the search radius is escalated until at least one pair is found.
    Returns nan for an empty cell or when nothing is found by `max_scale`
    (an isolated atom in a very large cell).
    """
    from ase.data import covalent_radii

    if len(atoms) == 0:
        return float('nan')
    rc = covalent_radii[atoms.numbers]
    scale = float(start_scale)
    while scale <= max_scale:
        try:
            i, j, _S, d = _bonded_pairs(atoms, scale, rc)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return float('nan')
        if len(i):
            return float(np.min(d / (rc[i] + rc[j])))
        scale *= 2.0
    return float('nan')


class _PerStructure:
    """
    Accumulates one row per structure.

    Everything here is derived from quantities the main loop has already
    computed, except the geometric descriptors and the exact contact ratio.
    Nothing in this class feeds back into the v1 metrics, so the numbers the v1
    report prints are byte-identical whether or not collection is enabled.
    """

    def __init__(self, save_per_atom=False, save_per_atom_forces=None):
        self.save_per_atom = bool(save_per_atom)
        self.save_per_atom_forces = (SAVE_PER_ATOM_FORCES
                                     if save_per_atom_forces is None
                                     else bool(save_per_atom_forces))
        # v2.1 ADDITION: raw per-atom force error, unbinned.
        # _af_n carries one entry per CALL to add(), 0 when the structure had
        # no forces, so the offsets stay aligned with the per-structure rows
        # even when some structures contribute no atoms.
        self._af_d = []
        self._af_ref = []
        self._af_z = []
        self._af_n = []
        self._c = {k: [] for k in (
            'index', 'sid', 'natoms', 'n_elements', 'elements', 'domain',
            'dim', 'n_fragments', 'min_cn', 'packing', 'volume',
            'volume_per_atom', 'contact_ratio',
            'ref_f_max', 'ref_f_mean', 'ref_f_rms',
            'ref_energy', 'ref_energy_per_atom', 'ref_stress_norm',
            'd_energy', 'abs_d_energy', 'abs_d_energy_per_atom',
            'f_mae_comp', 'f_rmse_comp', 'f_mae_norm', 'f_rmse_norm',
            'f_max_norm', 'f_sumsq',
            's_err_norm', 's_mae_comp9', 's_rmse_comp9', 's_mae_voigt6',
            'e_finite', 'f_finite', 's_finite')}
        self._pa_err = []
        self._pa_ref = []
        self._pa_frame = []
        self._pa_z = []
        # pooled per-atom histograms, per domain: 'comp' over the 3N
        # components, 'norm' over the N per-atom error vector lengths
        self._h = {}

    def _hist_add(self, domain, comp, norm):
        nb = PERATOM_HIST_BINS
        lo, hi = PERATOM_HIST_LO, PERATOM_HIST_HI
        for dk in ({'all', domain} if domain else {'all'}):
            h = self._h.setdefault(
                dk, {'comp': np.zeros(nb, dtype=np.int64),
                     'norm': np.zeros(nb, dtype=np.int64),
                     'n_nonfinite': 0})
            for key, v in (('comp', comp), ('norm', norm)):
                fin = np.isfinite(v)
                if not fin.all():
                    h['n_nonfinite'] += int((~fin).sum())
                vv = v[fin]
                if vv.size == 0:
                    continue
                lg = np.log10(np.maximum(vv, 10.0 ** lo))
                idx = np.clip(((lg - lo) / (hi - lo) * nb).astype(np.int64),
                              0, nb - 1)
                np.add.at(h[key], idx, 1)

    def hists(self):
        return self._h

    @staticmethod
    def _sid(atoms, frame_idx):
        for key in ('matpes_id', 'structure_id', 'sid', 'id'):
            if key in atoms.info:
                return str(atoms.info[key])
        return 'frame_%d' % frame_idx

    def add(self, frame_idx, ref_atoms, n_at, domain, desc, d_e,
            f_diff, f_norms, s_diff, f_ref, s_ref):
        c = self._c
        c['index'].append(frame_idx)
        c['sid'].append(self._sid(ref_atoms, frame_idx))
        c['natoms'].append(n_at)
        syms = sorted(set(ref_atoms.get_chemical_symbols()))
        c['n_elements'].append(len(syms))
        c['elements'].append(''.join(syms))
        c['domain'].append(domain)

        c['dim'].append(-1 if desc['dim'] is None else int(desc['dim']))
        c['n_fragments'].append(-1 if desc['n_fragments'] is None
                                else int(desc['n_fragments']))
        c['min_cn'].append(-1 if desc['min_cn'] is None
                           else int(desc['min_cn']))
        c['packing'].append(float(desc['packing']))
        c['volume'].append(float(desc['volume']))
        c['volume_per_atom'].append(float(desc['volume_per_atom']))
        c['contact_ratio'].append(
            min_contact_ratio_full(ref_atoms) if SAVE_TRUE_CONTACT_RATIO
            else float('nan'))

        # --- reference magnitudes (what the split tables are built on) -----
        if f_ref is None:
            c['ref_f_max'].append(float('nan'))
            c['ref_f_mean'].append(float('nan'))
            c['ref_f_rms'].append(float('nan'))
        else:
            rn = np.linalg.norm(np.asarray(f_ref, dtype=np.float64), axis=1)
            c['ref_f_max'].append(float(rn.max()) if rn.size else float('nan'))
            c['ref_f_mean'].append(float(rn.mean()) if rn.size
                                   else float('nan'))
            c['ref_f_rms'].append(float(np.sqrt((rn ** 2).mean())) if rn.size
                                  else float('nan'))
        e_ref = get_ref_energy(ref_atoms)
        c['ref_energy'].append(float('nan') if e_ref is None else float(e_ref))
        c['ref_energy_per_atom'].append(
            float('nan') if e_ref is None or not n_at else float(e_ref) / n_at)
        c['ref_stress_norm'].append(
            float('nan') if s_ref is None
            else float(np.linalg.norm(np.asarray(s_ref, dtype=np.float64))))

        # --- energy --------------------------------------------------------
        if d_e is None:
            c['d_energy'].append(float('nan'))
            c['abs_d_energy'].append(float('nan'))
            c['abs_d_energy_per_atom'].append(float('nan'))
            c['e_finite'].append(False)
        else:
            c['d_energy'].append(float(d_e))
            c['abs_d_energy'].append(float(abs(d_e)))
            c['abs_d_energy_per_atom'].append(
                float(abs(d_e) / n_at) if n_at else float('nan'))
            c['e_finite'].append(bool(np.isfinite(d_e)))

        # --- forces --------------------------------------------------------
        if f_diff is None or f_norms is None:
            for k in ('f_mae_comp', 'f_rmse_comp', 'f_mae_norm', 'f_rmse_norm',
                      'f_max_norm', 'f_sumsq'):
                c[k].append(float('nan'))
            c['f_finite'].append(False)
            if self.save_per_atom_forces:
                self._af_n.append(0)        # contributes no atoms
        else:
            ad = np.abs(f_diff)
            c['f_mae_comp'].append(float(ad.mean()))
            c['f_rmse_comp'].append(float(np.sqrt(np.square(f_diff).mean())))
            c['f_mae_norm'].append(float(f_norms.mean()))
            c['f_rmse_norm'].append(float(np.sqrt(np.square(f_norms).mean())))
            c['f_max_norm'].append(float(f_norms.max()))
            c['f_sumsq'].append(float(np.square(f_norms).sum()))
            c['f_finite'].append(bool(np.isfinite(f_diff).all()))
            self._hist_add(domain, np.abs(f_diff).ravel() * 1000.0,
                           f_norms * 1000.0)
            rn = None
            if self.save_per_atom or self.save_per_atom_forces:
                rn = (np.linalg.norm(np.asarray(f_ref, dtype=np.float64),
                                     axis=1).astype(np.float32)
                      if f_ref is not None
                      else np.full(n_at, np.nan, dtype=np.float32))
            if self.save_per_atom:
                self._pa_err.append(f_norms.astype(np.float32))
                self._pa_ref.append(rn)
                self._pa_frame.append(
                    np.full(n_at, frame_idx, dtype=np.int32))
                self._pa_z.append(
                    ref_atoms.numbers.astype(np.int16))
            # v2.1 ADDITION: the signed component-wise error, from which the
            # norm, |dF_i,k|, MAE_F and every quantile all follow. Signed
            # rather than absolute because it is the same size and strictly
            # more: it also carries the sign/softening information.
            if self.save_per_atom_forces:
                self._af_d.append(np.asarray(f_diff, dtype=np.float32))
                self._af_ref.append(rn)
                self._af_z.append(ref_atoms.numbers.astype(np.int16))
                self._af_n.append(n_at)

        # --- stress --------------------------------------------------------
        if s_diff is None:
            for k in ('s_err_norm', 's_mae_comp9', 's_rmse_comp9',
                      's_mae_voigt6'):
                c[k].append(float('nan'))
            c['s_finite'].append(False)
        else:
            c['s_err_norm'].append(float(np.linalg.norm(s_diff)))
            c['s_mae_comp9'].append(float(np.abs(s_diff).mean()))
            c['s_rmse_comp9'].append(float(np.sqrt(np.square(s_diff).mean())))
            v6 = np.array([s_diff[0, 0], s_diff[1, 1], s_diff[2, 2],
                           s_diff[1, 2], s_diff[0, 2], s_diff[0, 1]])
            c['s_mae_voigt6'].append(float(np.abs(v6).mean()))
            c['s_finite'].append(bool(np.isfinite(s_diff).all()))

    def finish(self):
        ints = {'index': np.int64, 'natoms': np.int64,
                'n_elements': np.int64, 'dim': np.int64,
                'n_fragments': np.int64, 'min_cn': np.int64}
        out = {}
        for k, v in self._c.items():
            if k in ints:
                out[k] = np.asarray(v, dtype=ints[k])
            elif k in ('sid', 'elements', 'domain'):
                out[k] = np.asarray(v, dtype=object).astype('U')
            elif k in ('e_finite', 'f_finite', 's_finite'):
                out[k] = np.asarray(v, dtype=bool)
            else:
                out[k] = np.asarray(v, dtype=np.float64)
        return out

    def per_atom_arrays(self):
        if not self.save_per_atom or not self._pa_err:
            return None
        return {'err_norm': np.concatenate(self._pa_err),
                'ref_norm': np.concatenate(self._pa_ref),
                'frame': np.concatenate(self._pa_frame),
                'Z': np.concatenate(self._pa_z)}

    def per_atom_force_arrays(self):
        """v2.1 ADDITION: the raw per-atom force error, unbinned.

        struct_offset is CSR-style and has one more entry than there are
        structures: the atoms of per-structure row j are
        [struct_offset[j]:struct_offset[j+1]], which is empty for a structure
        that carried no forces. That makes the join to per_structure.npz
        positional and exact, with no searchsorted and no reliance on frame
        ordering; 'frame' is stored as well so the join can also be made on
        the 'index' column.
        """
        if not self.save_per_atom_forces or not self._af_d:
            return None
        counts = np.asarray(self._af_n, dtype=np.int64)
        d = np.concatenate(self._af_d)
        idx = np.asarray(self._c['index'], dtype=np.int32)
        return {'d_force': d,
                'err_norm': np.linalg.norm(
                    d.astype(np.float64), axis=1).astype(np.float32),
                'ref_norm': np.concatenate(self._af_ref),
                'Z': np.concatenate(self._af_z),
                'frame': np.repeat(idx, counts),
                'struct_offset': np.concatenate(
                    ([0], np.cumsum(counts))).astype(np.int64)}


# ---------------------------------------------------------------------------
# Weighted robust statistics
# ---------------------------------------------------------------------------
# Non-finite errors are mapped to +inf rather than dropped, so a divergent
# model is ranked last instead of being silently improved by having its worst
# frames removed -- the same policy the v1 report follows for its sums.
def _worst_fill(v):
    v = np.array(v, dtype=np.float64, copy=True)
    v[~np.isfinite(v)] = np.inf
    return v


def _wquantile(v, w, q):
    """Weighted quantile; v may contain +inf."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = w > 0
    if not ok.any():
        return float('nan')
    v, w = v[ok], w[ok]
    o = np.argsort(v, kind='stable')
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float('nan')
    return float(v[o][min(int(np.searchsorted(cw, q * cw[-1])), len(v) - 1)])


def _wmean(v, w):
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = w > 0
    if not ok.any():
        return float('nan')
    return float(np.average(v[ok], weights=w[ok]))


def _wtrimmed_mean(v, w, frac):
    """Mean after dropping the worst `frac` of the WEIGHT (upper tail only)."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = w > 0
    if not ok.any():
        return float('nan')
    v, w = v[ok], w[ok]
    o = np.argsort(v, kind='stable')
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    keep = cw <= (1.0 - frac) * cw[-1]
    if not keep.any():
        keep[0] = True
    return float(np.average(v[keep], weights=w[keep]))


def _wwinsor_mean(v, w, cap, floor=None):
    """
    Weighted mean after clipping at a FIXED absolute cap (and optional floor).

    Unlike a trimmed mean, the cap is the same number for every model, so no
    model gains from having a more extreme tail. See the long note by
    TRIM_FRACTION for the measurement that motivates this.
    """
    v = np.asarray(v, dtype=np.float64)
    lo = -np.inf if floor is None else floor
    return _wmean(np.clip(v, lo, cap), w)


def _pooled_cap(arrays, weights, q=WINSOR_QUANTILE):
    """The common clipping cap: quantile q of every model's values pooled."""
    v = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays])
    w = np.concatenate([np.asarray(x, dtype=np.float64) for x in weights])
    fin = np.isfinite(v)
    if not fin.any():
        return float('nan')
    return _wquantile(v[fin], w[fin], q)


def _wgeomean(v, w, floor=1e-9):
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = (w > 0) & np.isfinite(v)
    if not ok.any():
        return float('nan')
    return float(np.exp(np.average(np.log(np.maximum(v[ok], floor)),
                                   weights=w[ok])))


# ---------------------------------------------------------------------------
# The paper tables
# ---------------------------------------------------------------------------
_DOMAIN_ORDER = ('all', 'bulk', 'non_bulk', 'overcompressed')
_DOMAIN_LABEL = {'all': 'ALL', 'bulk': 'BULK', 'non_bulk': 'NON-BULK',
                 'overcompressed': 'OVERCOMPRESSED'}


def _domain_masks(rows):
    """{domain key: boolean mask}. 'all' is always present and is everything."""
    dom = rows['domain']
    out = {'all': np.ones(len(dom), dtype=bool)}
    for k in _DOMAIN_ORDER[1:]:
        m = dom == k
        if m.any():
            out[k] = m
    return out


_QUANTITIES = (
    ('f_mae_comp', 'natoms', 1000.0, 'meV/A',
     'force MAE per structure (component convention)'),
    ('f_rmse_comp', 'natoms', 1000.0, 'meV/A',
     'force RMSE per structure (component convention)'),
    ('f_max_norm', 'natoms', 1000.0, 'meV/A',
     'largest per-atom force error in the structure'),
    ('abs_d_energy_per_atom', None, 1000.0, 'meV/atom',
     'energy error per atom'),
    ('s_mae_voigt6', None, 1000.0, 'meV/A^3',
     'stress MAE per structure (Voigt-6)'),
    ('s_err_norm', None, 1000.0, 'meV/A^3',
     'stress error Frobenius norm per structure'),
)


def _stat_block(v, w, quantiles=PAPER_QUANTILES, trim=TRIM_FRACTION):
    vv = _worst_fill(v)
    out = {'n': int(np.isfinite(np.asarray(v, dtype=np.float64)).sum()),
           'n_nonfinite': int((~np.isfinite(
               np.asarray(v, dtype=np.float64))).sum()),
           'mean_pooled': _wmean(vv, w),
           'trimmed_mean': _wtrimmed_mean(vv, w, trim),
           'geometric_mean': _wgeomean(vv, w)}
    for q in quantiles:
        out['q%g' % (100 * q)] = _wquantile(vv, w, q)
    return out


def _tbl_robust(rows, masks):
    res = {}
    for key, wcol, scale, unit, name in _QUANTITIES:
        if key not in rows or not np.isfinite(rows[key]).any():
            continue
        res[key] = {'unit': unit, 'scale': scale, 'name': name,
                    'weight': wcol or 'structure', 'domains': {}}
        for dk, m in masks.items():
            w = (rows[wcol][m].astype(np.float64) if wcol
                 else np.ones(int(m.sum()), dtype=np.float64))
            res[key]['domains'][dk] = _stat_block(rows[key][m], w)
    return res


def _print_robust(res):
    print('=' * 72)
    print('PAPER TABLE 1 -- ROBUST ACCURACY (per-structure error distribution)')
    print('  The pooled mean is the v1 MAE. It is printed last on purpose: on')
    print('  MatPES it is set by a handful of structures (see PAPER TABLE 3),')
    print('  so quote the median or the trimmed mean and give the seed noise')
    print('  floor from --aggregate next to it.')
    qs = ['q%g' % (100 * q) for q in PAPER_QUANTILES]
    for key, blk in res.items():
        print('-' * 72)
        print('  %s   [%s, weight: %s]'
              % (blk['name'], blk['unit'], blk['weight']))
        head = '  %-16s %9s' % ('domain', 'n')
        for q in qs:
            head += ' %10s' % q
        head += ' %11s %11s %11s' % ('trim%d%%' % round(100 * TRIM_FRACTION),
                                     'geo-mean', 'pooled mean')
        print(head)
        for dk in _DOMAIN_ORDER:
            if dk not in blk['domains']:
                continue
            s = blk['domains'][dk]
            line = '  %-16s %9d' % (_DOMAIN_LABEL[dk], s['n'])
            for q in qs:
                line += ' %10.2f' % (s[q] * blk['scale'])
            line += ' %11.2f %11.2f %11.2f' % (
                s['trimmed_mean'] * blk['scale'],
                s['geometric_mean'] * blk['scale'],
                s['mean_pooled'] * blk['scale'])
            print(line)
            if s['n_nonfinite']:
                print('  %-16s %s' % ('', '(%d structures with a non-finite '
                                          'error, counted as +inf)'
                                      % s['n_nonfinite']))


def _hist_quantile(h, q):
    """Quantile of a log-spaced histogram, in meV/A."""
    tot = h.sum()
    if tot <= 0:
        return float('nan')
    nb = len(h)
    centres = (PERATOM_HIST_LO + (np.arange(nb) + 0.5)
               * (PERATOM_HIST_HI - PERATOM_HIST_LO) / nb)
    k = int(np.searchsorted(np.cumsum(h), q * tot))
    return float(10.0 ** centres[min(k, nb - 1)])


def _tbl_peratom(hists, quantiles=PAPER_QUANTILES):
    """
    Pooled per-atom error quantiles, per domain.

    No inner average: every atom of every structure in the domain is one
    sample. This is robust to a single bad atom in a way the per-structure
    statistics of Table 1 are not, and is reported beside them so the reader
    can see the conclusion does not depend on which unit is chosen.

    Note that pooling atoms weights a structure by its atom count, exactly as
    the atom-weighted per-structure statistics do, so the two differ ONLY in
    whether a mean is taken inside each structure first.
    """
    res = {'bins': PERATOM_HIST_BINS, 'lo': PERATOM_HIST_LO,
           'hi': PERATOM_HIST_HI, 'unit': 'meV/A',
           'resolution_percent': 100.0 * (10.0 ** ((PERATOM_HIST_HI
                                                    - PERATOM_HIST_LO)
                                                   / PERATOM_HIST_BINS) - 1.0),
           'domains': {}}
    for dk, h in hists.items():
        entry = {'n_atoms': int(h['norm'].sum()),
                 'n_components': int(h['comp'].sum()),
                 'n_nonfinite': int(h['n_nonfinite'])}
        for conv in ('comp', 'norm'):
            for q in quantiles:
                entry['%s_q%g' % (conv, 100 * q)] = _hist_quantile(h[conv], q)
        res['domains'][dk] = entry
    return res


def _print_peratom(res):
    print('=' * 72)
    print('PAPER TABLE 1b -- POOLED PER-ATOM ERROR DISTRIBUTION')
    print('  Every atom is one sample; no averaging inside a structure. Table')
    print('  1 first takes a mean over each structure\'s atoms, which one bad')
    print('  atom can inflate; this table cannot be moved that way. Pooling')
    print('  atoms weights big structures more, exactly as the atom-weighted')
    print('  rows of Table 1 do, so the two differ only in that inner mean.')
    print('  Quantiles are read off a log-spaced histogram, exact to %.2f%%.'
          % res['resolution_percent'])
    qs = ['q%g' % (100 * q) for q in PAPER_QUANTILES]
    for conv, lab in (('comp', 'per component  |dF_i,k|  (MACE RMSE_F '
                               'convention)'),
                      ('norm', 'per atom       ||dF_i||  (error vector '
                               'length)')):
        print('-' * 72)
        print('  %s   [meV/A]' % lab)
        head = '  %-16s %12s' % ('domain', 'n samples')
        for q in qs:
            head += ' %10s' % q
        print(head)
        for dk in _DOMAIN_ORDER:
            if dk not in res['domains']:
                continue
            e = res['domains'][dk]
            n = e['n_components'] if conv == 'comp' else e['n_atoms']
            line = '  %-16s %12d' % (_DOMAIN_LABEL[dk], n)
            for q in qs:
                line += ' %10.2f' % e['%s_%s' % (conv, q)]
            print(line)
            if e['n_nonfinite']:
                print('  %-16s (%d non-finite values excluded from this table '
                      'only)' % ('', e['n_nonfinite']))


def _tbl_stability(rows, masks):
    res = {'force': {'thresholds': list(STABILITY_FORCE_THRESHOLDS),
                     'unit': 'eV/A', 'key': 'f_mae_comp', 'domains': {}},
           'energy': {'thresholds': list(STABILITY_ENERGY_THRESHOLDS),
                      'unit': 'eV/atom', 'key': 'abs_d_energy_per_atom',
                      'domains': {}}}
    for what, spec in res.items():
        v_all = rows[spec['key']]
        for dk, m in masks.items():
            v = _worst_fill(v_all[m])
            spec['domains'][dk] = {
                'n': int(m.sum()),
                'counts': [int((v > t).sum()) for t in spec['thresholds']],
                'fractions': [float((v > t).mean()) if v.size else float('nan')
                              for t in spec['thresholds']]}
    return res


def _print_stability(res):
    print('=' * 72)
    print('PAPER TABLE 2 -- STABILITY (structures above an error threshold)')
    print('  Reported separately from accuracy on purpose. A model can be')
    print('  typically accurate and occasionally divergent, or uniformly')
    print('  mediocre and never divergent; one number cannot say which.')
    for what, spec in res.items():
        print('-' * 72)
        print('  %s error per structure > threshold   [%s]'
              % (what, spec['unit']))
        head = '  %-16s %9s' % ('domain', 'n')
        for t in spec['thresholds']:
            head += ' %12s' % ('>%g' % t)
        print(head)
        for dk in _DOMAIN_ORDER:
            if dk not in spec['domains']:
                continue
            s = spec['domains'][dk]
            line = '  %-16s %9d' % (_DOMAIN_LABEL[dk], s['n'])
            for cnt, frac in zip(s['counts'], s['fractions']):
                line += ' %7d %4.2f%%' % (cnt, 100.0 * frac)
            print(line)


def _tbl_concentration(rows, masks, levels=(0.5, 0.9, 0.99, 0.999)):
    res = {'levels': list(levels), 'domains': {}}
    for dk, m in masks.items():
        sq = _worst_fill(rows['f_sumsq'][m])
        finite = np.isfinite(sq)
        tot = float(sq[finite].sum())
        entry = {'n': int(m.sum()), 'total_sumsq': tot,
                 'n_nonfinite': int((~finite).sum()), 'k': []}
        if tot > 0 and finite.any():
            srt = np.sort(sq[finite])[::-1]
            cum = np.cumsum(srt) / tot
            entry['k'] = [int(np.searchsorted(cum, L) + 1) for L in levels]
        res['domains'][dk] = entry
    return res


def _print_concentration(res):
    print('=' * 72)
    print('PAPER TABLE 3 -- ERROR CONCENTRATION')
    print('  Number of structures carrying a given share of the TOTAL SQUARED')
    print('  force error. This is the number that justifies not quoting the')
    print('  pooled RMSE: if 3 structures out of 323,096 carry half of it,')
    print('  the pooled RMSE is a measurement of those 3 structures.')
    head = '  %-16s %9s' % ('domain', 'n')
    for L in res['levels']:
        head += ' %12s' % ('%g%%' % (100 * L))
    print(head)
    for dk in _DOMAIN_ORDER:
        if dk not in res['domains']:
            continue
        s = res['domains'][dk]
        line = '  %-16s %9d' % (_DOMAIN_LABEL[dk], s['n'])
        for k in (s['k'] or [-1] * len(res['levels'])):
            line += ' %12s' % (k if k > 0 else '-')
        print(line)
        if s['n_nonfinite']:
            print('  %-16s (%d structures have a non-finite squared error and '
                  'are excluded from this table only)'
                  % ('', s['n_nonfinite']))


def _tbl_cdf(rows, masks, grid=CDF_FORCE_GRID):
    res = {'grid': list(grid), 'key': 'f_mae_comp', 'unit': 'eV/A',
           'domains': {}}
    for dk, m in masks.items():
        v = _worst_fill(rows['f_mae_comp'][m])
        w = rows['natoms'][m].astype(np.float64)
        tw = w.sum()
        res['domains'][dk] = {
            'n': int(m.sum()),
            'fraction_structures': [float((v <= t).mean()) if v.size
                                    else float('nan') for t in grid],
            'fraction_atom_weighted': [float(w[v <= t].sum() / tw)
                                       if tw > 0 else float('nan')
                                       for t in grid]}
    return res


def _print_cdf(res):
    print('=' * 72)
    print('PAPER TABLE 4 -- CUMULATIVE DISTRIBUTION of the per-structure '
          'force MAE')
    print('  Fraction of structures AT OR BELOW each threshold, atom-weighted.')
    print('  This is the table behind the CDF figure; the same numbers are')
    print('  written to cdf_forces_<domain>.csv for plotting. Draw the random')
    print('  seeds as a shaded band and the crossings become visible.')
    head = '  %-16s' % 'domain'
    for t in res['grid']:
        head += ' %8s' % ('%g' % (1000 * t))
    print(head + '   (thresholds in meV/A)')
    for dk in _DOMAIN_ORDER:
        if dk not in res['domains']:
            continue
        s = res['domains'][dk]
        line = '  %-16s' % _DOMAIN_LABEL[dk]
        for f in s['fraction_atom_weighted']:
            line += ' %7.2f%%' % (100.0 * f)
        print(line)


def _binned(rows, mask, binkey, edges, valkey='f_mae_comp',
            wcol='natoms'):
    v_all = _worst_fill(rows[valkey])
    b_all = np.asarray(rows[binkey], dtype=np.float64)
    out = []
    for lo, hi in zip(edges[:-1], list(edges[1:])):
        m = mask & (b_all >= lo) & (b_all < hi)
        out.append(_bin_entry('%g-%g' % (lo, hi), m, v_all, rows, wcol))
    m = mask & (b_all >= edges[-1])
    out.append(_bin_entry('>=%g' % edges[-1], m, v_all, rows, wcol))
    m = mask & ~np.isfinite(b_all)
    if m.any():
        out.append(_bin_entry('undefined', m, v_all, rows, wcol))
    return out


def _bin_entry(label, m, v_all, rows, wcol):
    n = int(m.sum())
    if n == 0:
        return {'bin': label, 'n': 0, 'median': float('nan'),
                'trimmed_mean': float('nan'), 'mean_pooled': float('nan')}
    w = (rows[wcol][m].astype(np.float64) if wcol
         else np.ones(n, dtype=np.float64))
    return {'bin': label, 'n': n,
            'median': _wquantile(v_all[m], w, 0.5),
            'trimmed_mean': _wtrimmed_mean(v_all[m], w, TRIM_FRACTION),
            'mean_pooled': _wmean(v_all[m], w)}


def _tbl_refbins(rows, masks, edges=REF_FORCE_BINS):
    return {'edges': list(edges), 'binkey': 'ref_f_max', 'unit': 'eV/A',
            'domains': {dk: _binned(rows, m, 'ref_f_max', list(edges))
                        for dk, m in masks.items()}}


def _print_refbins(res):
    print('=' * 72)
    print('PAPER TABLE 5 -- SPLIT BY REFERENCE FORCE MAGNITUDE')
    print('  Bins of max |F_ref| in the structure, so the split is computed')
    print('  from the DFT reference alone and cannot favour any model. Use')
    print('  this instead of, or beside, bulk/non-bulk: it is continuous and')
    print('  it separates near-equilibrium accuracy from strongly-driven')
    print('  configurations, which are different claims about a model.')
    for dk in _DOMAIN_ORDER:
        if dk not in res['domains']:
            continue
        print('-' * 72)
        print('  %s' % _DOMAIN_LABEL[dk])
        print('  %-14s %9s %12s %12s %12s'
              % ('max|F_ref|', 'n', 'median', 'trim mean', 'pooled mean'))
        for e in res['domains'][dk]:
            if e['n'] == 0:
                continue
            print('  %-14s %9d %12.2f %12.2f %12.2f'
                  % (e['bin'], e['n'], 1000 * e['median'],
                     1000 * e['trimmed_mean'], 1000 * e['mean_pooled']))


# ---------------------------------------------------------------------------
# Ceiling-relative analysis
# ---------------------------------------------------------------------------
def resolve_per_structure(path):
    """
    Accept either a per_structure.npz or the --paper_dir that contains one.

    Per-model runs are meant to keep their output next to the model, so
    aggregation has to be able to name those directories rather than force
    everything into one shared folder.
    """
    p = str(path)
    if os.path.isdir(p):
        cand = os.path.join(p, 'per_structure.npz')
        if not os.path.exists(cand):
            raise ValueError(
                '%s is a directory but contains no per_structure.npz. Pass '
                'either a --paper_dir from a previous run or the .npz file '
                'itself.' % p)
        return cand
    if not os.path.exists(p):
        raise ValueError('no such per-structure table: %s' % p)
    return p


def load_per_structure(path):
    """Read a per_structure.npz (or the directory holding one)."""
    p = resolve_per_structure(path)
    with np.load(p, allow_pickle=False) as z:
        rows = {k: z[k] for k in z.files if not k.startswith('__')}
    # Label priority: meta.json written by the run, else the directory name,
    # which is almost always what the user meant anyway.
    d = os.path.dirname(os.path.abspath(p))
    label = os.path.basename(d) or 'model'
    meta = os.path.join(d, 'meta.json')
    if os.path.exists(meta):
        import json
        try:
            label = json.load(open(meta)).get('label', label)
        except (ValueError, OSError):
            pass
    rows['__label__'] = label
    return rows


def align_to(rows, other):
    """
    Reorder `rows` onto `other`'s row order, matched on structure id.

    Returns (idx_rows, idx_other) such that rows[k][idx_rows] lines up with
    other[k][idx_other], element by element.

    THE FRAME SETS MUST BE IDENTICAL. Every comparison this script makes across
    models is PAIRED per structure -- the ceiling excess, the seed noise floor,
    the conditional profile -- and a paired statistic computed on two different
    frame sets is silently wrong, not noisy. So a mismatch is a hard error and
    never a quiet intersection: dropping the frames one model happens to be
    missing would change which structures the comparison runs on, which is
    exactly the selection effect the whole design is built to avoid.

    Differing ORDER is fine and is fixed here. Differing CONTENT is not. The
    atom counts are re-checked after matching, because equal ids with unequal
    atom counts means the tables came from different reference files.
    """
    a, b = rows['sid'], other['sid']
    if len(a) == len(b) and np.array_equal(a, b):
        ia = ib = np.arange(len(a))
    else:
        sa, sb = set(a.tolist()), set(b.tolist())
        if sa != sb:
            only_a = sorted(sa - sb)[:5]
            only_b = sorted(sb - sa)[:5]
            raise ValueError(
                'The two per-structure tables do not cover the same '
                'structures: %d ids only in the first (e.g. %s), %d only in '
                'the second (e.g. %s). Every cross-model number this script '
                'prints is paired per structure, so this cannot be papered '
                'over by intersecting -- re-run the evaluation of both models '
                'on the same reference file, with FRACTION = 1.0.'
                % (len(sa - sb), only_a, len(sb - sa), only_b))
        if len(a) != len(a.tolist()) or len(sa) != len(a):
            raise ValueError('Duplicate structure ids in a per-structure '
                             'table; ids must be unique.')
        pos = {sid: i for i, sid in enumerate(b)}
        ia = np.arange(len(a))
        ib = np.array([pos[sid] for sid in a])
    if not np.array_equal(rows['natoms'][ia], other['natoms'][ib]):
        raise ValueError(
            'Structure ids match but atom counts do not: the two '
            'per-structure tables are not from the same reference file.')
    return ia, ib


def _tbl_ceiling(rows, ceiling, key='f_mae_comp', wcol='natoms'):
    """
    Paired comparison against a CEILING MODEL trained on the full dataset.

    "Reference" in this script always means the DFT ground truth. The ceiling
    model is a separate prediction scored against that same DFT reference, so
    its error is not zero. Passing the ceiling's own table as both the model
    and the ceiling gives an excess of exactly 0, which is a useful check.

    Three numbers per domain:
      excess_median   median over structures of (e_model - e_ceiling). This is
                      the headline: pairing on the same structure cancels the
                      structure's intrinsic difficulty, which is what drops the
                      noise floor far enough for a few-meV/A effect to be
                      resolvable at all.
      ratio_geomean   geometric mean of e_model / e_ceiling, i.e. "how many
                      times the full-data model's error".
      profile         the model's own error, binned by the CEILING's error.

    On the profile: binning selects on the ceiling's error, so the CEILING's
    own column inside a bin is biased optimistic in low bins and pessimistic in
    high bins (regression to the mean) and must not be read as its skill there.
    The candidate models are unaffected -- none of them defined the bins -- so
    comparing candidates within a bin is sound. Every bin is printed; nothing
    is discarded.
    """
    ia, ib = align_to(rows, ceiling)
    sub = {k: v[ia] for k, v in rows.items() if isinstance(v, np.ndarray)
           and v.shape[:1] == rows['sid'].shape}
    cel = _worst_fill(ceiling[key][ib])
    mine = _worst_fill(sub[key])
    masks = _domain_masks(sub)
    res = {'key': key, 'n_matched': int(len(ia)), 'unit': 'eV/A',
           'ceiling_label': ceiling.get('__label__', 'ceiling'),
           'bins': list(CEILING_BINS), 'domains': {}}
    for dk, m in masks.items():
        w = sub[wcol][m].astype(np.float64) if wcol else None
        w = w if w is not None else np.ones(int(m.sum()))
        d = mine[m] - cel[m]
        fin = np.isfinite(d)
        ratio = np.full(int(m.sum()), np.nan)
        good = np.isfinite(mine[m]) & np.isfinite(cel[m]) & (cel[m] > 0)
        ratio[good] = mine[m][good] / cel[m][good]
        entry = {
            'n': int(m.sum()),
            'excess_median': (_wquantile(d[fin], w[fin], 0.5)
                              if fin.any() else float('nan')),
            'excess_q25': (_wquantile(d[fin], w[fin], 0.25)
                           if fin.any() else float('nan')),
            'excess_q75': (_wquantile(d[fin], w[fin], 0.75)
                           if fin.any() else float('nan')),
            'excess_trimmed_mean': (_wtrimmed_mean(d[fin], w[fin],
                                                   TRIM_FRACTION)
                                    if fin.any() else float('nan')),
            'ratio_geomean': _wgeomean(ratio[good], w[good]),
            'ratio_median': (_wquantile(ratio[good], w[good], 0.5)
                             if good.any() else float('nan')),
            'fraction_better_than_ceiling': (float(np.average(
                (d[fin] < 0).astype(float), weights=w[fin]))
                if fin.any() else float('nan')),
            'profile': [],
        }
        # profile: bin on the ceiling's error
        for lo, hi in zip(list(CEILING_BINS)[:-1], list(CEILING_BINS)[1:]):
            sel = m & (cel >= lo) & (cel < hi)
            entry['profile'].append(
                _ceil_bin('%g-%g' % (lo, hi), sel, mine, cel, sub, wcol))
        sel = m & (cel >= CEILING_BINS[-1])
        entry['profile'].append(
            _ceil_bin('>=%g' % CEILING_BINS[-1], sel, mine, cel, sub, wcol))
        res['domains'][dk] = entry
    return res


def _ceil_bin(label, sel, mine, cel, sub, wcol):
    n = int(sel.sum())
    if n == 0:
        return {'bin': label, 'n': 0}
    w = sub[wcol][sel].astype(np.float64) if wcol else np.ones(n)
    return {'bin': label, 'n': n,
            'ceiling_median': _wquantile(cel[sel], w, 0.5),
            'model_median': _wquantile(mine[sel], w, 0.5),
            'excess_median': _wquantile(mine[sel] - cel[sel], w, 0.5)}


# ---------------------------------------------------------------------------
# v2.1 ADDITION: per-atom statistics inside the ceiling bins
# ---------------------------------------------------------------------------
# The bins are unchanged -- they are still cut on the CEILING model's own
# per-structure f_mae_comp, so every row lines up with the tables above. Only
# the statistic reported inside a bin changes: instead of the atom-weighted
# median of a per-structure mean, every atom of every structure in the bin is
# one sample and the statistic is taken over those atoms directly. That
# removes the inner per-structure average, which is the one step that lets a
# single bad atom in a small cell move a whole structure's number.
#
#   median_norm   median over the bin's atoms of ||dF_i||
#   mae_f         mean over the bin's atoms of |dF_i,k| across all 3N
#                 components -- MACE's MAE_F, restricted to those atoms
def _per_atom_forces_path(path):
    """Where the per-atom file lives, given it or the directory holding it."""
    p = str(path)
    if os.path.isdir(p):
        return os.path.join(p, 'per_atom_forces.npz')
    if os.path.basename(p) == 'per_structure.npz':
        return os.path.join(os.path.dirname(os.path.abspath(p)),
                            'per_atom_forces.npz')
    return p


def has_per_atom_forces(path):
    """True if the per-atom file exists. Does NOT read it: probing eight
    models by loading them costs a gigabyte of pointless I/O."""
    return os.path.exists(_per_atom_forces_path(path))


def check_per_atom_forces(pa, rows=None, where='per_atom_forces.npz'):
    """Raise unless the per-atom file is internally sound and, when `rows` is
    given, describes exactly the structures in that per-structure table.

    This check is the difference between a wrong answer and an error message.
    struct_offset is positional: entry j is assumed to describe row j of
    per_structure.npz. If the two files come from different runs -- one model
    re-evaluated and the other not, a file copied from the wrong directory --
    nothing downstream would notice. Atoms would simply be attributed to the
    wrong structures and every table would look completely normal.

    The atom-count test is the one that actually catches that: a structure
    must contribute either all of its atoms or (having had no forces) none.
    """
    for k in ('d_force', 'err_norm', 'struct_offset'):
        if k not in pa:
            raise ValueError('%s has no %r (has: %s)'
                             % (where, k, ', '.join(sorted(pa))))
    off = np.asarray(pa['struct_offset'])
    if off.ndim != 1 or off.size < 1 or off[0] != 0:
        raise ValueError('%s: struct_offset must be 1-D and start at 0' % where)
    counts = np.diff(off)
    if (counts < 0).any():
        raise ValueError('%s: struct_offset is not non-decreasing' % where)
    n_at = int(off[-1])
    for k in ('d_force', 'err_norm'):
        if len(pa[k]) != n_at:
            raise ValueError(
                '%s: struct_offset ends at %d but %s has %d rows'
                % (where, n_at, k, len(pa[k])))
    if np.ndim(pa['d_force']) != 2 or np.shape(pa['d_force'])[1] != 3:
        raise ValueError('%s: d_force must be (N, 3), got %s'
                         % (where, np.shape(pa['d_force'])))
    if rows is None:
        return pa
    nat = np.asarray(rows['natoms'], dtype=np.int64)
    if len(counts) != len(nat):
        raise ValueError(
            '%s describes %d structures but the per-structure table has %d. '
            'The two files are from different runs; re-evaluate the model so '
            'that per_structure.npz and per_atom_forces.npz are written '
            'together.' % (where, len(counts), len(nat)))
    bad = ~((counts == nat) | (counts == 0))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            '%s disagrees with the per-structure table on atom counts: '
            '%d structure(s) differ, first at row %d (sid %s), which has %d '
            'atoms but contributes %d. The two files are not from the same '
            'run.' % (where, int(bad.sum()), i, rows['sid'][i], nat[i],
                      counts[i]))
    return pa


def load_per_atom_forces(path, rows=None):
    """Read a per_atom_forces.npz, given it or the directory holding it.

    Returns None rather than raising when the file is ABSENT, so a run made
    before this file existed degrades to exactly the old output. A file that
    is present but inconsistent raises instead: silence is only safe when
    there is nothing to be silent about.
    """
    p = _per_atom_forces_path(path)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        pa = {k: z[k] for k in z.files}
    return check_per_atom_forces(pa, rows, p)


def _peratom_view(pa, own_order=None, n_rows=None):
    """Per-atom arrays plus, for each atom, the row it belongs to.

    own_order[j] is the position IN THIS MODEL'S OWN TABLE of the structure
    that sits at position j of the common (base) order; passing None means the
    two orders already agree. The returned 'row' array is expressed in the
    COMMON order, so a structure mask built once can be applied to every
    model's atoms without rebuilding it.
    """
    off = pa['struct_offset'].astype(np.int64)
    counts = np.diff(off)
    own_row = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    if own_order is None:
        row = own_row
    else:
        # Belt and braces. An own_order shorter than the file would leave
        # rows unmapped at -1, and -1 is a legal index in numpy: those atoms
        # would be folded into the LAST structure and every number would come
        # out plausible and wrong. Fail instead.
        own_order = np.asarray(own_order, dtype=np.int64)
        if len(own_order) != len(counts):
            raise ValueError(
                'per-atom file describes %d structures but %d were matched; '
                'the per-atom and per-structure tables are not from the same '
                'run.' % (len(counts), len(own_order)))
        own2base = np.empty(len(counts), dtype=np.int64)
        own2base.fill(-1)
        own2base[own_order] = np.arange(len(own_order), dtype=np.int64)
        if (own2base < 0).any():
            raise ValueError(
                'per-atom file has %d structure(s) that no matched row points '
                'at; the tables are not from the same run.'
                % int((own2base < 0).sum()))
        row = own2base[own_row]
    d = pa['d_force']
    return {
        'row': row,
        # Non-finite norms become +inf, the same convention _worst_fill uses
        # everywhere else, so a divergent atom is ranked worst by the median
        # instead of being dropped.
        'err_norm': _worst_fill(pa['err_norm'].astype(np.float64)),
        # Mean of |dF_i,k| over this atom's own three components. Every atom
        # has exactly three, so the unweighted mean of this over a set of
        # atoms is identical to MAE_F over all their components.
        'comp_abs_mean': np.abs(d.astype(np.float64)).mean(axis=1),
        'n_rows': len(counts) if n_rows is None else n_rows,
    }


def _peratom_bin_stats(view, sel_struct):
    """median ||dF_i|| and MAE_F over the atoms of the selected structures."""
    m = sel_struct[view['row']]
    n = int(m.sum())
    if n == 0:
        return {'n_atoms': 0, 'median_norm': float('nan'),
                'mae_f': float('nan')}
    e = view['err_norm'][m]
    # _wquantile with unit weights, not np.median: every other quantile in
    # this script is an inverse-CDF quantile with no interpolation between
    # the two middle samples, and mixing the two conventions inside one
    # report would make rows disagree at the third digit for no reason.
    return {'n_atoms': n,
            'median_norm': _wquantile(e, np.ones(n), 0.5),
            'mae_f': float(view['comp_abs_mean'][m].mean())}


def _ceiling_bin_masks(cel, mask, edges=CEILING_BINS):
    """[(label, boolean structure mask)] for one domain -- the same cuts the
    per-structure profile uses, so the new rows align with the old ones."""
    out = []
    e = list(edges)
    for lo, hi in zip(e[:-1], e[1:]):
        out.append(('%g-%g' % (1000 * lo, 1000 * hi),
                    mask & (cel >= lo) & (cel < hi)))
    out.append(('>=%g' % (1000 * e[-1]), mask & (cel >= e[-1])))
    return out


# ---------------------------------------------------------------------------
# v4 ADDITION: ceiling profiles binned on the ceiling's own PER-ATOM error,
# absolute and relative
# ---------------------------------------------------------------------------
# Why this needs machinery the v2.1 block did not.
#
# Every per-atom statistic in v2.1 is taken over ONE model's own atoms: the bin
# is a set of STRUCTURES, each model selects its own atoms from it, and the
# median is computed independently. No atom of model A ever has to be matched
# to the corresponding atom of model B, so the STRUCTURE permutation returned
# by align_to() is all that is needed.
#
# Here the bin membership is fixed by the CEILING's per-atom error and then
# applied to another model's atoms. That is an atom-level pairing, and getting
# it wrong would not raise: atoms would be attributed to the wrong bins and
# every row would look entirely plausible. _peratom_base_perm builds the
# permutation from struct_offset, and the caller then checks the atomic numbers
# and the reference force magnitudes agree after it has been applied -- two
# quantities that are stored per atom precisely so that this can be checked
# rather than assumed.
def _peratom_base_perm(pa, own_order, base_counts, where='per_atom_forces.npz'):
    """Row permutation putting a model's per-atom arrays into the COMMON order.

    own_order[j] is the position of common row j in THIS model's own table,
    i.e. exactly what align_to() returns. Atoms of a structure are stored
    contiguously, so the permutation follows from struct_offset alone -- but
    only if every structure contributes the same number of atoms in both
    files, which is checked here rather than assumed.
    """
    off = np.asarray(pa['struct_offset'], dtype=np.int64)
    counts = np.diff(off)
    own_order = np.asarray(own_order, dtype=np.int64)
    if len(own_order) != len(counts):
        raise ValueError(
            '%s describes %d structures but %d were matched; the per-atom and '
            'per-structure tables are not from the same run.'
            % (where, len(counts), len(own_order)))
    base_counts = np.asarray(base_counts, dtype=np.int64)
    mine = counts[own_order]
    if not np.array_equal(mine, base_counts):
        bad = int(np.flatnonzero(mine != base_counts)[0])
        raise ValueError(
            '%s contributes %d atoms for the structure at common row %d, but '
            'the reference model contributes %d there. A structure that '
            'carried no forces in one run and all of them in another cannot '
            'be paired atom by atom.' % (where, int(mine[bad]), bad,
                                         int(base_counts[bad])))
    total = int(base_counts.sum())
    out_start = np.concatenate(([0], np.cumsum(base_counts)[:-1]))
    return (np.arange(total, dtype=np.int64)
            - np.repeat(out_start, base_counts)
            + np.repeat(off[own_order], base_counts))


def _atombin_masks(cv, mask, edges, label_scale=1.0):
    """[(label, atom mask)] cut on the CEILING's own per-atom value `cv`."""
    out = []
    e = list(edges)
    for lo, hi in zip(e[:-1], e[1:]):
        out.append(('%g-%g' % (label_scale * lo, label_scale * hi),
                    mask & (cv >= lo) & (cv < hi)))
    out.append(('>=%g' % (label_scale * e[-1]), mask & (cv >= e[-1])))
    return out


def _atombin_profiles(labs, jobs, ceiling_job, base_counts, struct_masks):
    """Both v4 per-atom-binned profiles, in one pass over the models.

    jobs         [(label, per-atom path, own_order, own per-structure table)]
    ceiling_job  the same 4-tuple for the ceiling
    base_counts  atoms per structure, in the COMMON row order
    struct_masks {domain: structure mask, COMMON row order}

    Returns {'norm': {domain: [row]}, 'rel': {domain: [row]}, 'info': {...}}.
    One model is held in memory at a time, as in the v2.1 block; only the
    ceiling's two arrays are kept for the whole pass, because every bin is cut
    on them.
    """
    base_counts = np.asarray(base_counts, dtype=np.int64)
    row = np.repeat(np.arange(len(base_counts), dtype=np.int64), base_counts)
    amask = {dk: m[row] for dk, m in struct_masks.items()}

    cpath, corder, ctab = ceiling_job[1], ceiling_job[2], ceiling_job[3]
    cpa = load_per_atom_forces(cpath, rows=ctab)
    cperm = _peratom_base_perm(cpa, corder, base_counts, cpath)
    cerr = _worst_fill(cpa['err_norm'].astype(np.float64))[cperm]
    cref = np.asarray(cpa['ref_norm'], dtype=np.float64)[cperm]
    cz = np.asarray(cpa['Z'])[cperm]
    del cpa

    # The relative table's population. A reference force of exactly zero is not
    # a small number, it is a site where rho does not exist; those atoms are
    # dropped and counted, never quietly folded in at some large value.
    keep = cref > REL_MIN_REF_FORCE
    crel = np.full(len(cerr), np.nan)
    crel[keep] = cerr[keep] / cref[keep]

    specs = {'norm': {}, 'rel': {}}
    for dk, m in amask.items():
        specs['norm'][dk] = _atombin_masks(cerr, m, PERATOM_CEILING_BINS,
                                           1000.0)
        specs['rel'][dk] = _atombin_masks(crel, m & keep, PERATOM_REL_BINS,
                                          1.0)

    def _cells(vnorm, vrel):
        got = {}
        for tag, v in (('norm', vnorm), ('rel', vrel)):
            got[tag] = {}
            for dk in specs[tag]:
                col = []
                for _lab, s in specs[tag][dk]:
                    n = int(s.sum())
                    col.append(_wquantile(v[s], np.ones(n), 0.5)
                               if n >= PERATOM_MIN_BIN_ATOMS else float('nan'))
                got[tag][dk] = col
        return got

    ceil_cells = _cells(cerr, crel)

    cells = {}
    for L, path, order, tab in jobs:
        pa = load_per_atom_forces(path, rows=tab)
        perm = _peratom_base_perm(pa, order, base_counts, path)
        if not np.array_equal(np.asarray(pa['Z'])[perm], cz):
            raise ValueError(
                'after matching, the per-atom file for %r does not describe '
                'the same atoms in the same order as the ceiling (the atomic '
                'numbers differ). The two evaluations did not read the same '
                'reference file, so their atoms cannot be paired.' % L)
        ref = np.asarray(pa['ref_norm'], dtype=np.float64)[perm]
        err = _worst_fill(pa['err_norm'].astype(np.float64))[perm]
        del pa
        if not np.allclose(ref, cref, rtol=1e-5, atol=1e-12):
            raise ValueError(
                'the reference force magnitudes stored for %r differ from the '
                'ceiling\'s after matching; the two evaluations did not read '
                'the same reference file.' % L)
        rel = np.full(len(err), np.nan)
        rel[keep] = err[keep] / cref[keep]
        cells[L] = _cells(err, rel)
        del err, rel, ref

    res = {'norm': {}, 'rel': {},
           'info': {'min_ref_force': float(REL_MIN_REF_FORCE),
                    'n_atoms': int(len(cerr)),
                    'n_excluded_rel': int((~keep).sum()),
                    'n_zero_ref': int((cref <= 0).sum()),
                    'domains': {}}}
    for dk in amask:
        res['info']['domains'][dk] = {
            'n_atoms': int(amask[dk].sum()),
            'n_excluded_rel': int((amask[dk] & ~keep).sum())}
    for tag in ('norm', 'rel'):
        for dk in specs[tag]:
            n_dom = int((amask[dk] if tag == 'norm'
                         else (amask[dk] & keep)).sum())
            rws = []
            for bi, (blab, s) in enumerate(specs[tag][dk]):
                n = int(s.sum())
                if n < PERATOM_MIN_BIN_ATOMS:
                    continue
                vals = [cells[L][tag][dk][bi] for L in labs]
                mean, sd, _ = _seed_stats(vals, labs)
                best = labs[int(np.argmin([v if np.isfinite(v) else np.inf
                                           for v in vals]))]
                rws.append({'bin': blab, 'n': n,
                            'pct': 100.0 * n / max(n_dom, 1),
                            'ceiling_value': ceil_cells[tag][dk][bi],
                            'values': {L: v for L, v in zip(labs, vals)},
                            'seed_mean': mean, 'seed_sd': sd, 'best': best})
            res[tag][dk] = rws
    return res


def _print_atombin(res, labs, tag, scale, vfmt, binunit, title, note):
    """One of the two v4 per-atom-binned profiles."""
    print('-' * 72)
    print('  %s' % title)
    for line in note:
        print('  %s' % line)
    info = res['info']
    if tag == 'rel':
        print('  %d of %d atoms have no defined relative error and are '
              'excluded' % (info['n_excluded_rel'], info['n_atoms']))
        print('  (||F_ref,i|| <= %g eV/A; %d of them are exactly zero by '
              'symmetry).' % (info['min_ref_force'], info['n_zero_ref']))
    for dk in _DOMAIN_ORDER:
        if dk not in res[tag]:
            continue
        d = info['domains'].get(dk, {})
        extra = ('' if tag == 'norm'
                 else ', %d excluded' % d.get('n_excluded_rel', 0))
        print('  %s  (%d atoms%s)' % (_DOMAIN_LABEL[dk],
                                      d.get('n_atoms', 0), extra))
        head = '    %-20s %11s %9s %9s' % ('ceiling bin [%s]' % binunit,
                                           'n atoms', '% of dom', 'ceil')
        for L in labs:
            head += ' %10s' % L[:10]
        head += ' %10s %10s' % ('seed mean', 'seed sd')
        for L in labs:
            if not _is_seed_label(L):
                head += ' %9s' % ('sd(%s)' % L[:5])
        head += ' %8s' % 'best'
        print(head)
        rws = res[tag][dk]
        if not rws:
            print('    (no bin with at least %d atoms)' % PERATOM_MIN_BIN_ATOMS)
            continue
        for r in rws:
            line = ('    %-20s %11d %8.2f%% ' % (r['bin'], r['n'], r['pct'])
                    + (vfmt % (r['ceiling_value'] * scale)))
            for L in labs:
                line += ' ' + (vfmt % (r['values'][L] * scale))
            line += ' ' + (vfmt % (r['seed_mean'] * scale))
            line += ' ' + (vfmt % (r['seed_sd'] * scale))
            for L in labs:
                if not _is_seed_label(L):
                    line += ' %9s' % _fmt_sigma(r['values'][L], r['seed_mean'],
                                                r['seed_sd'])
            line += ' %8s' % r['best'][:8]
            print(line)
        n_shown = sum(r['n'] for r in rws)
        n_dom = (info['domains'][dk]['n_atoms'] if tag == 'norm'
                 else info['domains'][dk]['n_atoms']
                 - info['domains'][dk]['n_excluded_rel'])
        if n_dom - n_shown:
            print('    (%d atoms in bin(s) with fewer than %d members, '
                  'omitted above)' % (n_dom - n_shown, PERATOM_MIN_BIN_ATOMS))
        cov = {}
        for r in rws:
            cov[r['best']] = cov.get(r['best'], 0) + r['n']
        parts = ['%s %.2f%%' % (L, 100.0 * cov[L] / max(n_dom, 1))
                 for L in labs if cov.get(L)]
        print('    bins where each model is best cover: %s' % ', '.join(parts))


def _print_ceiling(res):
    print('=' * 72)
    print('PAPER TABLE 6 -- RELATIVE TO THE FULL-DATA CEILING (%s)'
          % res['ceiling_label'])
    print('  Paired per structure, %d structures matched by id.'
          % res['n_matched'])
    print('  excess = this model\'s per-structure force MAE minus the')
    print('  ceiling\'s, on the SAME structure. Nothing is excluded: where the')
    print('  ceiling itself fails, the excess is small or negative and the')
    print('  median simply ignores it, which is the effect a hard in/out cut')
    print('  was reaching for -- without a threshold to defend.')
    print('  %-16s %9s %11s %11s %11s %11s %10s'
          % ('domain', 'n', 'excess med', 'excess q25', 'excess q75',
             'ratio geo', '% better'))
    for dk in _DOMAIN_ORDER:
        if dk not in res['domains']:
            continue
        s = res['domains'][dk]
        print('  %-16s %9d %11.2f %11.2f %11.2f %11.4f %9.2f%%'
              % (_DOMAIN_LABEL[dk], s['n'], 1000 * s['excess_median'],
                 1000 * s['excess_q25'], 1000 * s['excess_q75'],
                 s['ratio_geomean'],
                 100.0 * s['fraction_better_than_ceiling']))
    print('-' * 72)
    print('  CEILING-CONDITIONAL PROFILE (all bins shown, nothing discarded)')
    print('  Compare CANDIDATE MODELS to each other within a bin. Do NOT read')
    print('  the ceiling column as the ceiling\'s skill in that bin: the bins')
    print('  were selected on its own error, so it regresses to the mean and')
    print('  is pinned inside the bin edges by construction -- it is a LABEL')
    print('  for the regime, not a score. With one model there are no seed')
    print('  replicas here, so no noise floor can be shown; run --aggregate')
    print('  over all models to get the seed mean and sd per bin.')
    for dk in _DOMAIN_ORDER:
        if dk not in res['domains']:
            continue
        print('  %s' % _DOMAIN_LABEL[dk])
        print('    %-14s %9s %8s %13s %13s %13s'
              % ('ceiling bin', 'n', '%', 'ceiling med', 'model med',
                 'excess med'))
        tot = max(res['domains'][dk]['n'], 1)
        for e in res['domains'][dk]['profile']:
            if e['n'] == 0:
                continue
            print('    %-14s %9d %7.2f%% %13.2f %13.2f %13.2f'
                  % (e['bin'], e['n'], 100.0 * e['n'] / tot,
                     1000 * e['ceiling_median'], 1000 * e['model_median'],
                     1000 * e['excess_median']))


def _tbl_ceiling_peratom(rows, ceiling, pa, ceiling_pa, key='f_mae_comp'):
    """v2.1 ADDITION: the ceiling-conditional profile, scored per ATOM.

    Same structures, same bins, same order as _tbl_ceiling's profile; the two
    can be read side by side row for row.
    """
    ia, ib = align_to(rows, ceiling)
    sub = {k: v[ia] for k, v in rows.items() if isinstance(v, np.ndarray)
           and v.shape[:1] == rows['sid'].shape}
    cel = _worst_fill(ceiling[key][ib])
    masks = _domain_masks(sub)
    mine_view = _peratom_view(pa, own_order=ia)
    ceil_view = _peratom_view(ceiling_pa, own_order=ib)
    res = {'key': key, 'unit': 'meV/A', 'bins': list(CEILING_BINS),
           'ceiling_label': ceiling.get('__label__', 'ceiling'),
           'domains': {}}
    for dk, m in masks.items():
        prof = []
        for lab, sel in _ceiling_bin_masks(cel, m):
            n = int(sel.sum())
            if n == 0:
                continue
            mine = _peratom_bin_stats(mine_view, sel)
            ceil = _peratom_bin_stats(ceil_view, sel)
            prof.append({'bin': lab, 'n': n, 'n_atoms': mine['n_atoms'],
                         'model_median_norm': mine['median_norm'],
                         'ceiling_median_norm': ceil['median_norm'],
                         'model_mae_f': mine['mae_f'],
                         'ceiling_mae_f': ceil['mae_f']})
        res['domains'][dk] = {'n': int(m.sum()), 'profile': prof}
    return res


def _print_ceiling_peratom(res):
    """PAPER TABLE 6b/6c -- the same profile, scored per atom."""
    for tag, mk, ck, title, note in (
            ('6b', 'model_median_norm', 'ceiling_median_norm',
             'MEDIAN PER-ATOM ERROR ||dF_i|| IN EACH CEILING BIN',
             'Every atom of every structure in the bin is one sample; no '
             'mean is taken inside a structure first.'),
            ('6c', 'model_mae_f', 'ceiling_mae_f',
             "MACE MAE_F OVER THE ATOMS OF EACH CEILING BIN",
             'Mean of |dF_i,k| over all 3N components of the bin\'s atoms '
             '-- MACE\'s MAE_F, restricted to those atoms.')):
        print('=' * 72)
        print('PAPER TABLE %s -- %s' % (tag, title))
        print('  %s' % note)
        print('  Bins are cut on the CEILING\'s per-structure error, exactly '
              'as in Table 6,')
        print('  so these rows line up with it one for one. The ceiling '
              'column labels the')
        print('  regime and is not its skill there (it selected the bins).')
        for dk in _DOMAIN_ORDER:
            if dk not in res['domains']:
                continue
            d = res['domains'][dk]
            print('  %s' % _DOMAIN_LABEL[dk])
            print('    %-14s %9s %12s %8s %13s %13s %11s'
                  % ('ceiling bin', 'n struct', 'n atoms', '% of dom',
                     'ceiling', 'model', 'difference'))
            tot = max(d['n'], 1)
            for e in d['profile']:
                print('    %-14s %9d %12d %7.2f%% %13.2f %13.2f %11.2f'
                      % (e['bin'], e['n'], e['n_atoms'],
                         100.0 * e['n'] / tot,
                         1000 * e[ck], 1000 * e[mk],
                         1000 * (e[mk] - e[ck])))
            if not d['profile']:
                print('    (no structures)')
        print('  "difference" is model minus ceiling of the two POOLED '
              'numbers above,')
        print('  not a per-atom paired median -- the two are not the same '
              'statistic.')


# ---------------------------------------------------------------------------
# v5 ADDITION: the blow-up census -- how many ATOMS exceed a force-error
# threshold on the NORM convention
# ---------------------------------------------------------------------------
# Conventions, all of them ones the rest of this script already follows:
#
#   * the quantity is ||dF_i||, the length of the atom's own error vector, NOT
#     MACE's component-average MAE_F and NOT any per-structure mean. An atom is
#     one sample. (If you need the same census in the 'rms_component' units a
#     MACE log prints, divide the threshold by sqrt(3): the two orderings are
#     identical, see the FORCE_ERROR_CUTOFF_ON note at the top of this file.)
#
#   * the test is STRICT, count = #{i : ||dF_i|| > t}, exactly as
#     _tbl_stability tests its per-structure values.
#
#   * a non-finite ||dF_i|| has already been mapped to +inf by _peratom_view,
#     so it is counted above EVERY threshold rather than dropped: a divergent
#     model may not be improved by having its worst atoms vanish. How many
#     atoms that is, per model and per domain, is reported alongside, so the
#     finite and the non-finite parts of a count can be separated again.
#
#   * the denominator is the domain's own atom count as the per-atom file
#     reports it, NOT sum(natoms) over the domain's structures: a structure
#     that carried no forces contributes no atoms, and the per-atom file is
#     the only thing that knows which ones those are.
#
# Structure masks are built once in the common (base) row order and applied to
# every model's atoms through view['row'], which _peratom_view has already
# expressed in that same order. A per-model rebuild is exactly where a paired
# comparison would drift onto different structures.
def _blowup_domain_masks(base_masks):
    """[(domain key, structure mask)] for the census, in print order.

    The usual domains, plus their union without the overcompressed screen.
    That union is taken from the masks rather than as 'all' minus
    'overcompressed' so that it remains the population it claims to be even if
    the domains stop partitioning the set.
    """
    out = [(dk, base_masks[dk]) for dk in _DOMAIN_ORDER if dk in base_masks]
    if 'bulk' in base_masks and 'non_bulk' in base_masks:
        out.append(('bulk_plus_non_bulk',
                    base_masks['bulk'] | base_masks['non_bulk']))
    return out


_BLOWUP_DOMAIN_LABEL = dict(_DOMAIN_LABEL)
_BLOWUP_DOMAIN_LABEL['bulk_plus_non_bulk'] = 'BULK + NON-BULK'


def _peratom_blowup_counts(view, raw_err, dom_masks,
                           thresholds=PERATOM_BLOWUP_THRESHOLDS):
    """One model's census: {domain: {n_atoms, n_nonfinite, counts}}.

    `raw_err` is the file's own err_norm, before _worst_fill mapped the
    non-finite entries to +inf; it is used only to count them, never to
    threshold, so the counts keep the +inf convention.
    """
    e = view['err_norm']
    row = view['row']
    nonfin = ~np.isfinite(np.asarray(raw_err, dtype=np.float64))
    out = {}
    for dk, m in dom_masks:
        sel = m[row]
        ee = e[sel]
        out[dk] = {'n_atoms': int(ee.size),
                   'n_nonfinite': int(nonfin[sel].sum()),
                   'counts': [int((ee > t).sum()) for t in thresholds]}
    return out


def _tbl_peratom_blowup(labs, pa_jobs, dom_masks,
                        thresholds=PERATOM_BLOWUP_THRESHOLDS):
    """The census for every model. One per-atom file in memory at a time.

    pa_jobs is [(column label, path, matched order, per-structure table)], the
    same tuple the per-atom profile blocks build; '__ceiling__' may be one of
    the labels and is carried through as an extra column without entering the
    seed statistics.
    """
    per_model = {}
    for L, q, it, t in pa_jobs:
        pa = load_per_atom_forces(q, rows=t)
        view = _peratom_view(pa, own_order=it)
        per_model[L] = _peratom_blowup_counts(view, pa['err_norm'], dom_masks,
                                              thresholds)
        del view, pa
    res = {'quantity': 'per-atom force error norm ||dF_i||', 'unit': 'eV/A',
           'test': 'strict, > threshold',
           'nonfinite_policy': 'mapped to +inf, counted above every threshold',
           'thresholds': list(thresholds), 'columns': list(per_model),
           'domains': {}}
    for dk, _m in dom_masks:
        na = {L: per_model[L][dk]['n_atoms'] for L in per_model}
        ent = {'n_atoms': na,
               'n_nonfinite': {L: per_model[L][dk]['n_nonfinite']
                               for L in per_model},
               'thresholds': {}}
        for i, t in enumerate(thresholds):
            c = {L: per_model[L][dk]['counts'][i] for L in per_model}
            mean, sd, _ = _seed_stats([float(c[L]) for L in labs], labs)
            ent['thresholds']['>%g' % t] = {
                'counts': c,
                'fractions': {L: (c[L] / na[L] if na[L] else float('nan'))
                              for L in per_model},
                'seed_mean': mean, 'seed_sd': sd}
        res['domains'][dk] = ent
    return res


def _print_peratom_blowup(res, labs, cols, col_names):
    """cols are the keys into the table, col_names what to print for them."""
    print('=' * 72)
    print('BLOW-UP CENSUS -- ATOMS whose own force error norm ||dF_i|| '
          'exceeds a threshold')
    print('  One ATOM is one sample. No per-structure mean is taken first, so '
          'this is a')
    print('  count of divergent atoms and not, as the STABILITY table above '
          'is, a count')
    print('  of structures whose component-averaged MAE came out large. The '
          'two answer')
    print('  different questions and neither can be derived from the other.')
    print('  Non-finite errors count as +inf, i.e. above every threshold; '
          'they are')
    print('  reported separately on the "non-finite" line so they can be '
          'taken back out.')
    for dk, ent in res['domains'].items():
        print('-' * 72)
        na = ent['n_atoms']
        print('  %s  (n_atoms=%d)'
              % (_BLOWUP_DOMAIN_LABEL.get(dk, dk.upper()),
                 na.get(cols[0], 0)))
        if len({na[L] for L in cols}) > 1:
            print('  WARNING: the models disagree on this domain\'s atom '
                  'count: %s.'
                  % ', '.join('%s %d' % (n, na[L])
                              for L, n in zip(cols, col_names)))
            print('  Percentages below are each taken over that model\'s own '
                  'total.')
        head = '  %-18s' % 'threshold eV/A'
        for n in col_names:
            head += ' %10s' % n[:10]
        head += ' %12s %8s' % ('seed mean', 'seed sd')
        print(head)
        for t in res['thresholds']:
            e = ent['thresholds']['>%g' % t]
            line = '  %-18s' % ('>%g' % t)
            for L in cols:
                line += ' %10d' % e['counts'][L]
            line += ' %12.1f %8.1f' % (e['seed_mean'], e['seed_sd'])
            print(line)
        print('  -- the same counts as a % of the domain\'s atoms --')
        for t in res['thresholds']:
            e = ent['thresholds']['>%g' % t]
            line = '  %-18s' % ('>%g' % t)
            for L in cols:
                line += ' %10.4f' % (100.0 * e['fractions'][L])
            print(line)
        nf = ent['n_nonfinite']
        if any(nf[L] for L in cols):
            line = '  %-18s' % 'non-finite'
            for L in cols:
                line += ' %10d' % nf[L]
            print(line)
            print('    (included in every count above; a model with '
                  'non-finite errors has')
            print('     no finite blow-up threshold and its counts are floors, '
                  'not measurements)')


# ---------------------------------------------------------------------------
# v5 ADDITION: the pooled PER-ATOM force MAE, both conventions, per domain
# ---------------------------------------------------------------------------
# See the "PAPER TABLE 1c" note at the top of this file for why both
# conventions are printed together and why neither converts into the other.
#
# The arithmetic is an exact identity, not an approximation. Writing a_s for
# the atoms of structure s and N_s = |a_s|,
#
#     f_mae_comp_s = (1 / 3N_s) * sum_{i in a_s} sum_k |dF_i,k|
#     f_mae_norm_s = (1 /  N_s) * sum_{i in a_s} ||dF_i||
#
# so sum_s N_s * f_mae_*_s / sum_s N_s collapses to the mean over every atom
# (or every component) of the domain: the per-structure average cancels
# against the weight. That is the same identity the module docstring states
# for the v1 pooled MAE, here reused for the norm convention as well.
#
# Consequences worth knowing:
#   * the table needs only per_structure.npz. It is produced whether or not
#     per_atom_forces.npz exists, unlike every other per-atom table here;
#   * the 'comp' row must reproduce, exactly, the 'pooled_mean_MAE' row that
#     Table 1 already prints for f_mae_comp. That is checked and reported, and
#     it is the cheapest possible guard against a domain mask or a row
#     alignment having drifted between the two tables;
#   * the denominator is sum(natoms) over the domain's structures. A structure
#     that carried no forces has a non-finite f_mae_*, which _worst_fill maps
#     to +inf, so it poisons the mean rather than being dropped -- the same
#     policy, and the same visible failure, as the existing pooled row.
_PERATOM_MAE_KEYS = (
    ('norm', 'f_mae_norm', 'mean over the atoms of ||dF_i||'),
    ('comp', 'f_mae_comp',
     "mean over all 3N components of |dF_i,k| (MACE's MAE_F)"),
)


def _peratom_force_mae_table(labs, jobs, dom_masks, wcol='natoms'):
    """Pooled per-atom force MAE per domain, both conventions.

    jobs is [(column label, per-structure table in the COMMON row order)]; a
    ceiling may be one of them and is carried as an extra column without
    entering the seed statistics.
    """
    for tag, key, _d in _PERATOM_MAE_KEYS:
        absent = [L for L, t in jobs if key not in t]
        if absent:
            raise ValueError(
                'no %r column in the per-structure table(s) for %s. Only '
                'tables written by v2 or later carry both force conventions; '
                're-run those evaluations to get the %s row.'
                % (key, ', '.join(absent), tag))
    res = {'unit': 'eV/A',
           'conventions': {tag: desc for tag, _k, desc in _PERATOM_MAE_KEYS},
           'columns': [L for L, _t in jobs], 'domains': {}}
    for dk, m in dom_masks:
        ent = {'n_atoms': int(jobs[0][1][wcol][m].sum()),
               'n_structures': int(m.sum())}
        for tag, key, _desc in _PERATOM_MAE_KEYS:
            vals = {L: _wmean(_worst_fill(t[key][m]),
                              t[wcol][m].astype(np.float64))
                    for L, t in jobs}
            ordered = [vals[L] for L in labs]
            mean, sd, _ = _seed_stats(ordered, labs)
            ent[tag] = {
                'values': vals, 'seed_mean': mean, 'seed_sd': sd,
                'sigma': {L: ((vals[L] - mean) / sd
                              if np.isfinite(vals[L]) and np.isfinite(sd)
                              and sd > 0 else float('nan'))
                          for L in vals},
                'rank_order': [labs[i] for i in np.argsort(
                    [v if np.isfinite(v) else np.inf for v in ordered])]}
        res['domains'][dk] = ent
    return res


def _peratom_force_mae_direct(pa_jobs, dom_masks):
    """The same two numbers taken straight from the per-atom arrays.

    An independent route to the table above: it counts atoms as the per-atom
    file reports them instead of trusting sum(natoms), and it never touches
    the per-structure force columns. Used only to CHECK the table, never to
    produce it, so a missing or unreadable per-atom file costs the check and
    not the table. One model is held in memory at a time.
    """
    out = {}
    for L, q, it, t in pa_jobs:
        view = _peratom_view(load_per_atom_forces(q, rows=t), own_order=it)
        out[L] = {}
        for dk, m in dom_masks:
            sel = m[view['row']]
            n = int(sel.sum())
            out[L][dk] = {
                'n_atoms': n,
                'norm': float(view['err_norm'][sel].mean()) if n
                else float('nan'),
                # every atom has exactly three components, so the unweighted
                # mean of each atom's own mean|dF_i,k| is MAE_F over all of
                # them -- see the note in _peratom_view.
                'comp': float(view['comp_abs_mean'][sel].mean()) if n
                else float('nan')}
        del view
    return out


def _print_peratom_force_mae(res, labs, cols, col_names, direct=None,
                             pooled_row=None):
    print('=' * 72)
    print('PER-ATOM FORCE MAE -- pooled over a domain\'s atoms, both '
          'conventions')
    print('  norm : %s' % res['conventions']['norm'])
    print('  comp : %s' % res['conventions']['comp'])
    print('  These are two different measurements of the same errors, not two '
          'units of')
    print('  one. Their ratio is bounded by [1, 3] and is 2 only for '
          'isotropic errors,')
    print('  so a number quoted in one convention may never be compared with '
          'one in the')
    print('  other. Report which you used.')
    for dk, ent in res['domains'].items():
        print('-' * 72)
        print('  %s  (%d structures, %d atoms)'
              % (_BLOWUP_DOMAIN_LABEL.get(dk, dk.upper()),
                 ent['n_structures'], ent['n_atoms']))
        head = '  %-18s' % 'convention'
        for n in col_names:
            head += ' %10s' % n[:10]
        head += ' %12s %8s' % ('seed mean', 'seed sd')
        for L, n in zip(cols, col_names):
            if L not in labs or not _is_seed_label(L):
                head += ' %10s' % ('sd(%s)' % n[:5])
        print(head)
        for tag, _key, _desc in _PERATOM_MAE_KEYS:
            e = ent[tag]
            line = '  %-18s' % ('%s [meV/A]' % tag)
            for L in cols:
                line += ' %10.2f' % (1000 * e['values'][L])
            line += ' %12.2f %8.2f' % (1000 * e['seed_mean'],
                                       1000 * e['seed_sd'])
            for L in cols:
                if L not in labs or not _is_seed_label(L):
                    line += ' %10s' % _fmt_sigma(e['values'][L],
                                                 e['seed_mean'], e['seed_sd'])
            print(line)
        print('    ranking by norm: %s' % ' < '.join(ent['norm']['rank_order']))
        ratio = [ent['norm']['values'][L] / ent['comp']['values'][L]
                 for L in cols if np.isfinite(ent['comp']['values'][L])
                 and ent['comp']['values'][L] > 0]
        if ratio:
            print('    norm/comp ratio over the models: %.3f - %.3f '
                  '(bounded by [1, 3]; 2 iff isotropic)'
                  % (min(ratio), max(ratio)))
        # Guard 1: the comp row must BE the pooled_mean_MAE row of Table 1.
        if pooled_row and dk in pooled_row:
            # n_cmp counts the comparisons actually made. It is NOT implied by
            # `worst`: an exact match leaves worst at 0.0, and testing that
            # alone would report a perfect agreement as "nothing to compare".
            worst, who, n_cmp = 0.0, None, 0
            for L in cols:
                if L not in pooled_row[dk]:
                    continue
                a, b = ent['comp']['values'][L], pooled_row[dk][L]
                if np.isfinite(a) and np.isfinite(b) and abs(b) > 0:
                    n_cmp += 1
                    r = abs(a - b) / abs(b)
                    if r > worst:
                        worst, who = r, L
            if not n_cmp:
                print('    (no overlap with Table 1 to check against)')
            elif worst <= 1e-12:
                print('    comp row reproduces Table 1\'s pooled_mean_MAE for '
                      'f_mae_comp (%d models, max rel. dev. %.2g)'
                      % (n_cmp, worst))
            else:
                print('    WARNING: comp row differs from Table 1\'s '
                      'pooled_mean_MAE by up to %.3g (worst: %s). The two '
                      'tables are not seeing the same rows.' % (worst, who))
        # Guard 2: the same numbers rebuilt from the raw per-atom arrays.
        if direct:
            # n_cmp for the same reason as above: exact agreement is the
            # expected outcome here, and must not read as "nothing compared".
            worst, who, what, n_cmp = 0.0, None, None, 0
            for L in cols:
                for tag, _k, _d in _PERATOM_MAE_KEYS:
                    a, b = ent[tag]['values'][L], direct[L][dk][tag]
                    if np.isfinite(a) and np.isfinite(b) and abs(b) > 0:
                        n_cmp += 1
                        r = abs(a - b) / abs(b)
                        if r > worst:
                            worst, who, what = r, L, tag
            na = {L: direct[L][dk]['n_atoms'] for L in cols}
            if len(set(na.values())) > 1 or ent['n_atoms'] not in set(na.values()):
                print('    NOTE: sum(natoms)=%d but the per-atom files hold %s '
                      'atoms here; structures that carried no forces '
                      'contribute none.'
                      % (ent['n_atoms'],
                         '/'.join(str(v) for v in sorted(set(na.values())))))
            if not n_cmp:
                print('    (per-atom cross-check produced nothing comparable)')
            elif worst <= 1e-5:
                print('    both conventions reproduced from the raw per-atom '
                      'arrays (%d checks, max rel. dev. %.2g)'
                      % (n_cmp, worst))
            else:
                print('    WARNING: the per-atom arrays give a different %s '
                      'value for %s (rel. dev. %.3g). The per-structure force '
                      'columns and the per-atom file disagree.'
                      % (what, who, worst))


# ---------------------------------------------------------------------------
# Driver: print the paper sections and save everything
# ---------------------------------------------------------------------------
def _paper_report(rows, paper_dir=None, ceiling_path=None, model_label=None,
                  ref_path=None, pred_path=None, non_bulk_detector=None,
                  contact_ratio_cutoff=None, per_atom=None,
                  peratom_hists=None, per_atom_forces=None):
    if FRACTION != 1.0:
        print('')
        print('=' * 72)
        print('WARNING: FRACTION = %g, so these paper tables were computed on '
              'a random' % FRACTION)
        print('  subset of the test set. Set FRACTION = 1.0 before producing '
              'anything')
        print('  you intend to publish or to compare across models: every '
              'cross-model')
        print('  number this script prints is paired per structure, and the '
              'pooled')
        print('  RMSE does not subsample at all (see the module docstring).')
    masks = _domain_masks(rows)
    tables = {'robust': _tbl_robust(rows, masks),
              'stability': _tbl_stability(rows, masks),
              'concentration': _tbl_concentration(rows, masks),
              'cdf': _tbl_cdf(rows, masks),
              'ref_force_bins': _tbl_refbins(rows, masks)}
    if peratom_hists:
        tables['peratom'] = _tbl_peratom(peratom_hists)
    _print_robust(tables['robust'])
    if 'peratom' in tables:
        _print_peratom(tables['peratom'])
    _print_stability(tables['stability'])
    _print_concentration(tables['concentration'])
    _print_cdf(tables['cdf'])
    _print_refbins(tables['ref_force_bins'])

    if ceiling_path:
        try:
            ceiling = load_per_structure(ceiling_path)
            tables['ceiling'] = _tbl_ceiling(rows, ceiling)
            _print_ceiling(tables['ceiling'])
        except (OSError, ValueError, KeyError) as exc:
            print('=' * 72)
            print('PAPER TABLE 6 -- skipped: could not use --ceiling %s'
                  % ceiling_path)
            print('  %s' % exc)
        else:
            # v2.1 ADDITION. Purely additive: needs a per_atom_forces.npz for
            # THIS model and for the ceiling, and says so and moves on when
            # either is absent, leaving every table above untouched.
            ceiling_pa = load_per_atom_forces(ceiling_path, rows=ceiling)
            if per_atom_forces is not None:
                check_per_atom_forces(per_atom_forces, rows,
                                      'this run\'s per-atom arrays')
            if per_atom_forces is None or ceiling_pa is None:
                print('=' * 72)
                print('PAPER TABLES 6b/6c -- skipped: no per_atom_forces.npz '
                      'for %s'
                      % ('this run (--no_per_atom_forces?)'
                         if per_atom_forces is None else 'the ceiling model'))
                print('  Re-run with per-atom output enabled to get the '
                      'per-atom profile.')
            else:
                try:
                    tables['ceiling_peratom'] = _tbl_ceiling_peratom(
                        rows, ceiling, per_atom_forces, ceiling_pa)
                    _print_ceiling_peratom(tables['ceiling_peratom'])
                except (OSError, ValueError, KeyError) as exc:
                    print('=' * 72)
                    print('PAPER TABLES 6b/6c -- skipped: %s' % exc)

    if paper_dir:
        _save_paper(paper_dir, rows, tables, per_atom=per_atom,
                    peratom_hists=peratom_hists,
                    per_atom_forces=per_atom_forces,
                    meta={'label': model_label or 'model',
                          'reference_file': ref_path,
                          'prediction_file': pred_path,
                          'non_bulk_detector': non_bulk_detector,
                          'contact_ratio_cutoff': contact_ratio_cutoff,
                          'trim_fraction': TRIM_FRACTION,
                          'n_structures': int(len(rows['sid'])),
                          'ceiling_file': ceiling_path})
    return tables


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if np.isfinite(f) else str(f)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def _save_paper(out_dir, rows, tables, per_atom=None, meta=None,
                peratom_hists=None, per_atom_forces=None):
    import gzip
    import json

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, 'per_structure.npz'), **rows)
    with open(os.path.join(out_dir, 'tables.json'), 'w') as fh:
        json.dump(_jsonable(tables), fh, indent=1, sort_keys=True)
    if meta is not None:
        with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
            json.dump(_jsonable(meta), fh, indent=1, sort_keys=True)

    cols = [k for k in rows if isinstance(rows[k], np.ndarray)
            and rows[k].ndim == 1]
    order = ['sid', 'index', 'natoms', 'domain', 'elements'] + \
            [c for c in cols if c not in
             ('sid', 'index', 'natoms', 'domain', 'elements')]
    with gzip.open(os.path.join(out_dir, 'per_structure.csv.gz'), 'wt') as fh:
        fh.write(','.join(order) + '\n')
        vals = [rows[c] for c in order]
        for i in range(len(rows['sid'])):
            fh.write(','.join(
                ('%.8g' % v[i]) if v.dtype.kind == 'f' else str(v[i])
                for v in vals) + '\n')

    grid = tables['cdf']['grid']
    for dk, s in tables['cdf']['domains'].items():
        with open(os.path.join(out_dir, 'cdf_forces_%s.csv' % dk), 'w') as fh:
            fh.write('threshold_eV_per_A,threshold_meV_per_A,'
                     'fraction_structures,fraction_atom_weighted\n')
            for t, a, b in zip(grid, s['fraction_structures'],
                               s['fraction_atom_weighted']):
                fh.write('%.6g,%.6g,%.8g,%.8g\n' % (t, 1000 * t, a, b))

    if per_atom:
        np.savez_compressed(os.path.join(out_dir, 'per_atom.npz'), **per_atom)
    if per_atom_forces:
        np.savez_compressed(os.path.join(out_dir, 'per_atom_forces.npz'),
                            **per_atom_forces)
    if peratom_hists:
        flat = {'bins': np.array([PERATOM_HIST_BINS]),
                'lo': np.array([PERATOM_HIST_LO]),
                'hi': np.array([PERATOM_HIST_HI])}
        for dk, h in peratom_hists.items():
            flat['%s_comp' % dk] = h['comp']
            flat['%s_norm' % dk] = h['norm']
        np.savez_compressed(
            os.path.join(out_dir, 'peratom_hist.npz'), **flat)

    with open(os.path.join(out_dir, 'README.txt'), 'w') as fh:
        fh.write(_PAPER_README)
    print('=' * 72)
    print('Paper data written to %s/' % out_dir)
    for f in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, f)
        print('  %-28s %10.1f MB' % (f, os.path.getsize(p) / 1e6))
    print('  Aggregate several models by naming their paper directories:')
    print('    ./eval_with_report_efs_v2.py \\')
    print('        --aggregate /path/v14_paper /path/DIRECT_paper '
          '/path/rdm1_paper ... \\')
    print('        --ceiling   /path/OMAT0_paper')
    print('  (labels are read from each run\'s meta.json; --labels only '
          'overrides them)')


_PAPER_README = """\
Per-structure MatPES evaluation data
====================================
Written by eval_with_report_efs_v2.py. One row per structure; nothing is
aggregated, so any split or estimator can be recomputed offline without
re-running the model.

FILES
  per_structure.npz     all columns below, as numpy arrays
  per_structure.csv.gz  the same table as text
  tables.json           every number printed in the PAPER TABLE sections
  cdf_forces_*.csv      ready-to-plot CDF of the per-structure force MAE
  per_atom.npz          only with --save_per_atom: err_norm, ref_norm, frame, Z
  peratom_hist.npz      pooled per-atom error histograms, <domain>_comp and
                        <domain>_norm, log-spaced (see PERATOM_HIST_* in the
                        script for the grid)
  per_atom_forces.npz   RAW UNBINNED per-atom force error, one row per atom
                        (suppressed by --no_per_atom_forces):
                          d_force       (N,3) float32  signed dF = pred - ref,
                                                       eV/A
                          err_norm      (N,)  float32  ||dF_i||
                          ref_norm      (N,)  float32  ||F_ref,i||
                          Z             (N,)  int16    atomic number
                          frame         (N,)  int32    joins to the 'index'
                                                       column above
                          struct_offset (M+1,) int64   row j owns atoms
                                                       [off[j]:off[j+1]]
                        Everything per-atom follows from d_force: the norm is
                        sqrt((d_force**2).sum(1)), MACE's MAE_F over any set of
                        atoms is abs(d_force[sel]).mean(), and the continuous
                        unbinned CDF is sort(err_norm) against arange(N)/N --
                        no binning is applied anywhere.
  meta.json             model label, input files, settings used

IDENTIFICATION
  sid                   matpes_id when present, else frame_<index>
  index                 0-based frame index in the reference file
  natoms, n_elements, elements   elements is the sorted symbol string, e.g.
                        'CoLiMnO', so it can be grouped on directly
  domain                'bulk' | 'non_bulk' | 'overcompressed' | 'all'
                        (as assigned by this run's detector settings)

GEOMETRIC DESCRIPTORS (from the REFERENCE structure only, never the prediction)
  dim                   periodic dimensionality 0-3 of the bonded network
  n_fragments           connected components under the bond cutoff
  min_cn                smallest coordination number
  packing               sum of covalent-sphere volumes / cell volume; a density
                        index, NOT a packing fraction -- see the long note in
                        the script before using it as a classifier
  volume, volume_per_atom          A^3
  contact_ratio         EXACT minimum over all pairs and images of
                        d_ij / (R_cov,i + R_cov,j). 1.0 = touching, below ~0.4
                        = unphysically compressed reference data. Continuous,
                        so it can be re-thresholded offline.

REFERENCE MAGNITUDES (for model-independent splits)
  ref_f_max, ref_f_mean, ref_f_rms   over per-atom |F_ref|, eV/A
  ref_energy, ref_energy_per_atom    eV, eV/atom
  ref_stress_norm                    Frobenius norm of the reference stress

ERRORS (prediction minus reference; all in eV, eV/A, eV/A^3)
  d_energy              signed, eV
  abs_d_energy, abs_d_energy_per_atom
  f_mae_comp            mean |dF| over the structure's 3N components. This is
                        the per-structure form of MACE's RMSE_F quantity and is
                        the recommended headline error.
  f_rmse_comp           sqrt(mean dF^2) over the 3N components
  f_mae_norm            mean per-atom ||dF_i||
  f_rmse_norm           sqrt(mean ||dF_i||^2)
  f_max_norm            largest single-atom ||dF_i|| in the structure
  f_sumsq               sum_i ||dF_i||^2 -- the structure's contribution to the
                        pooled squared error, which is what the concentration
                        table is built from
  s_err_norm, s_mae_comp9, s_rmse_comp9, s_mae_voigt6
  e_finite, f_finite, s_finite       False if the model emitted inf/nan

CONVENTIONS THAT MATTER WHEN YOU RECOMPUTE
  * Weight forces by natoms to get an atom-weighted statistic; the pooled
    (atom-weighted) mean of f_mae_comp reproduces the v1 MAE exactly.
  * Non-finite errors are mapped to +inf, never dropped, so a divergent model
    ranks last rather than being improved by the exclusion.
  * Energy and stress are per-structure quantities: weight them equally, do
    not atom-weight them.
"""


# ---------------------------------------------------------------------------
# Multi-model aggregation: the actual paper tables
# ---------------------------------------------------------------------------
def _is_seed_label(label):
    low = str(label).lower()
    return any(low.startswith(p) for p in SEED_LABEL_PREFIXES)


def _seed_stats(values, labels):
    """(mean, sd, n) over the random-seed replicas only."""
    v = [values[i] for i, L in enumerate(labels)
         if _is_seed_label(L) and np.isfinite(values[i])]
    if len(v) < 2:
        return float('nan'), float('nan'), len(v)
    return float(np.mean(v)), float(np.std(v, ddof=1)), len(v)


def _fmt_sigma(x, mean, sd):
    if not (np.isfinite(x) and np.isfinite(mean) and np.isfinite(sd)) or sd <= 0:
        return '     -'
    return '%+6.1f' % ((x - mean) / sd)


# v3 ADDITION -- aggregation-only re-classification of the bulk/non-bulk split.
#
# The descriptors every detector in NON_BULK_CRITERIA reads (packing, dim,
# n_fragments, min_cn) are already columns of per_structure.npz, so the split
# can be recomputed from a finished evaluation without re-running any model.
# Only the PREDICATE changes: the descriptors themselves were built once, at
# NON_BULK_BOND_SCALE, during the evaluation, and cannot be rebuilt here --
# changing the bond scale still requires re-running the evaluation.
#
# Which descriptor columns each detector consumes. Kept next to the code that
# uses it rather than next to NON_BULK_CRITERIA so that every v3 change stays
# inside the aggregation section; it is only used for the sentinel check
# below, and the predicate is always handed all four columns regardless.
_DETECTOR_INPUTS = {
    'packing': ('packing',),
    'dim': ('dim',),
    'fragmented': ('n_fragments',),
    'dim_or_frag': ('dim', 'n_fragments'),
    'min_cn': ('min_cn',),
    'any': ('packing', 'dim', 'n_fragments', 'min_cn'),
}


def _rederive_domains(tables, detector, thr=NON_BULK_PACKING_THRESHOLD):
    """
    Relabel bulk / non-bulk in every loaded table from the stored descriptors.

    Applied identically to every model AND to the ceiling, so a paired
    comparison is never made across two different domain definitions.

    The predicate is the very NON_BULK_CRITERIA entry the evaluation path uses
    -- not a reimplementation of it -- and it is applied one row at a time,
    because those predicates combine their tests with Python's `or`, which
    does not vectorise. One pass over the descriptors is shared by all tables.

    'overcompressed' is left exactly as stored. That screen runs BEFORE the
    detector during evaluation, is independent of it, and its input
    (contact_ratio) is only saved when SAVE_TRUE_CONTACT_RATIO is on, so
    re-deriving it here would add a dependency for no gain.

    Returns (criterion label, number of rows relabelled, n_overcompressed).
    """
    label, _needs_graph, fn = resolve_non_bulk_detector(detector)
    cols = ('packing', 'dim', 'n_fragments', 'min_cn')

    for name, t in tables:
        for k in cols + ('domain',):
            if k not in t:
                raise ValueError(
                    'cannot re-derive the domain: %s has no %r column. Only '
                    'tables written by this script (v2 or later) carry the '
                    'geometric descriptors.' % (name, k))
    base = tables[0][1]

    # Every table must describe the same structures with the same descriptors;
    # otherwise they came from different reference files or different bond
    # scales and a common relabelling would be meaningless. align_to() has
    # already matched sid/natoms, but it does not look at the descriptors.
    for name, t in tables[1:]:
        for k in cols:
            if not np.array_equal(t[k], base[k]):
                raise ValueError(
                    'cannot re-derive the domain: %r has different %r values '
                    'than %r, so the tables were not built from the same '
                    'reference file with the same NON_BULK_BOND_SCALE.'
                    % (name, k, tables[0][0]))

    # -1 is the sentinel the capture writes when a descriptor was never
    # computed (an evaluation run with a detector that needs no bonded graph).
    # Re-deriving from it would silently mislabel every affected structure.
    for k in _DETECTOR_INPUTS[detector]:
        if k in ('dim', 'n_fragments', 'min_cn') and np.any(base[k] < 0):
            raise ValueError(
                'cannot re-derive the domain with detector %r: the stored %r '
                'column contains %d sentinel (-1) entries, i.e. the evaluation '
                'ran with a detector that never built the bonded graph. '
                'Re-run the evaluation with a graph-based --non_bulk_detector.'
                % (detector, k, int(np.sum(base[k] < 0))))

    old = base['domain']
    new = old.copy()
    keep = old == 'overcompressed'
    for r in np.nonzero(~keep)[0]:
        d = {'packing': float(base['packing'][r]),
             'dim': int(base['dim'][r]),
             'n_fragments': int(base['n_fragments'][r]),
             'min_cn': int(base['min_cn'][r])}
        new[r] = 'non_bulk' if bool(fn(d, thr)) else 'bulk'
    for _name, t in tables:
        t['domain'] = new.copy()
    return label, int(np.sum(new != old)), int(np.sum(keep))


def aggregate_models(paths, labels=None, out_dir='matpes_paper_tables',
                     ceiling_path=None, agg_non_bulk_detector=None):
    """
    Read several per_structure.npz files and print/save the paper tables.

    The random-seed replicas (labels starting with rdm/seed/random) define the
    NOISE FLOOR: for every metric the table prints their mean +- sd and, for
    every model, how many sd it sits from that mean. A difference smaller than
    the seed sd is not a result, whatever its sign, and stating this for the
    pooled MAE and RMSE is what lets you keep those columns in the paper
    honestly instead of omitting them.
    """
    import json

    tabs = [load_per_structure(p) for p in paths]
    if labels:
        if len(labels) != len(paths):
            raise ValueError('--labels must have one name per --aggregate file')
        labs = list(labels)
    else:
        labs = [t.get('__label__', 'model%d' % i) for i, t in enumerate(tabs)]
    dup = {L for L in labs if labs.count(L) > 1}
    if dup:
        raise ValueError(
            'duplicate model label(s) %s -- every model needs a distinct name, '
            'because the tables are keyed by it. Give --labels explicitly, or '
            'rename the --paper_dir of the clashing runs.' % sorted(dup))

    aligned = []
    # v2.1: row_order[i][j] = row of base structure j in model i's own table.
    # NOT named `order`: that name is reused below for label rankings.
    row_order = []
    base = tabs[0]
    for lab, t in zip(labs, tabs):
        try:
            _, it = align_to(base, t)
        except ValueError as exc:
            raise ValueError('model %r does not align with %r: %s'
                             % (lab, labs[0], exc)) from None
        aligned.append({k: v[it] for k, v in t.items()
                        if isinstance(v, np.ndarray)
                        and v.shape[:1] == t['sid'].shape})
        row_order.append(it)
        assert np.array_equal(aligned[-1]['sid'], base['sid'])

    ceiling = None
    if ceiling_path:
        c = load_per_structure(ceiling_path)
        # Reindex the ceiling INTO base order (ic tells us where each base row
        # lives in c), not the other way round: every array below is indexed by
        # the base masks.
        _, ic = align_to(aligned[0], c)
        ceiling_order = ic          # v2.1: same role as order[i], for the ceiling
        ceiling_tab = c             # v2.1: unreordered, to validate against
        ceiling = {k: v[ic] for k, v in c.items()
                   if isinstance(v, np.ndarray)
                   and v.shape[:1] == c['sid'].shape}
        ceiling['__label__'] = c.get('__label__', 'ceiling')
        assert np.array_equal(ceiling['sid'], aligned[0]['sid'])

    # v3 ADDITION. With no --aggregate_non_bulk_detector this block does not
    # run and the stored classification is used exactly as before.
    rederived = None
    if agg_non_bulk_detector is not None:
        relabel_tabs = list(zip(labs, aligned))
        if ceiling is not None:
            relabel_tabs.append((str(ceiling['__label__']) + ' (ceiling)',
                                 ceiling))
        crit, n_moved, n_over = _rederive_domains(relabel_tabs,
                                                  agg_non_bulk_detector)
        rederived = {'detector': agg_non_bulk_detector, 'criterion': crit,
                     'n_relabelled': n_moved, 'n_overcompressed_kept': n_over,
                     'n_tables': len(relabel_tabs),
                     'bond_scale_of_stored_descriptors': NON_BULK_BOND_SCALE}

    base_masks = _domain_masks(aligned[0])

    n_seeds = sum(1 for L in labs if _is_seed_label(L))
    print('')
    print('=' * 72)
    print('MatPES PAPER TABLES -- %d models, %d structures'
          % (len(labs), len(aligned[0]['sid'])))
    print('  models : %s' % ', '.join(labs))
    print('  seeds  : %d treated as random-seed replicas (noise floor)'
          % n_seeds)
    if ceiling:
        print('  ceiling: %s' % ceiling['__label__'])
    if rederived:
        print('  domains: RE-DERIVED HERE, detector %r (%s)'
              % (rederived['detector'], rederived['criterion']))
        print('           %d of %d structures relabelled against the split '
              'stored' % (rederived['n_relabelled'], len(aligned[0]['sid'])))
        print('           at evaluation time; applied to all %d tables, the '
              'ceiling included.' % rederived['n_tables'])
        print('           %d overcompressed structures left as stored. The '
              'descriptors' % rederived['n_overcompressed_kept'])
        print('           were built at NON_BULK_BOND_SCALE=%g and are reused '
              'as-is.' % NON_BULK_BOND_SCALE)
    if n_seeds < 2:
        print('  WARNING: fewer than 2 seed replicas -- no noise floor can be')
        print('  estimated, and no difference below can be called significant.')

    out = {'labels': labs, 'files': list(paths),
           'n_structures': int(len(aligned[0]['sid'])), 'metrics': {}}
    if rederived:
        out['rederived_domains'] = rederived

    # 'trimmed_mean' is kept because it is conventional, but it is printed
    # with a warning: see the note by TRIM_FRACTION. 'winsor_mean' is the
    # comparable version, clipped for every model at the same pooled cap.
    ESTIMATORS = (
        ('median', lambda v, w, cap: _wquantile(v, w, 0.5)),
        ('q75', lambda v, w, cap: _wquantile(v, w, 0.75)),
        ('q90', lambda v, w, cap: _wquantile(v, w, 0.90)),
        ('winsor_mean', lambda v, w, cap: _wwinsor_mean(v, w, cap)),
        ('trimmed_mean*', lambda v, w, cap: _wtrimmed_mean(v, w,
                                                           TRIM_FRACTION)),
        ('geometric_mean', lambda v, w, cap: _wgeomean(v, w)),
        ('pooled_mean_MAE', lambda v, w, cap: _wmean(v, w)),
    )

    for key, wcol, scale, unit, name in _QUANTITIES:
        if key not in aligned[0]:
            continue
        if not any(np.isfinite(a[key]).any() for a in aligned):
            continue
        out['metrics'][key] = {'unit': unit, 'name': name, 'domains': {}}
        print('=' * 72)
        print('%s   [%s]' % (name.upper(), unit))
        for dk in _DOMAIN_ORDER:
            if dk not in base_masks:
                continue
            m = base_masks[dk]
            print('-' * 72)
            print('  %s  (n=%d)' % (_DOMAIN_LABEL[dk], int(m.sum())))
            head = '  %-18s' % 'estimator'
            for L in labs:
                head += ' %10s' % L[:10]
            head += ' %12s %8s' % ('seed mean', 'seed sd')
            for L in labs:
                if not _is_seed_label(L):
                    head += ' %10s' % ('sd(%s)' % L[:5])
            print(head)
            dom = {}
            _ws = [(_worst_fill(a[key][m]),
                    (a[wcol][m].astype(np.float64) if wcol
                     else np.ones(int(m.sum())))) for a in aligned]
            cap = _pooled_cap([x[0] for x in _ws], [x[1] for x in _ws])
            for est_name, fn in ESTIMATORS:
                vals = []
                for vv, w in _ws:
                    vals.append(fn(vv, w, cap))
                mean, sd, _ = _seed_stats(vals, labs)
                line = '  %-18s' % est_name
                for v in vals:
                    line += ' %10.2f' % (v * scale)
                line += ' %12.2f %8.2f' % (mean * scale, sd * scale)
                for L, v in zip(labs, vals):
                    if not _is_seed_label(L):
                        line += ' %10s' % _fmt_sigma(v, mean, sd)
                print(line)
                order = [labs[i] for i in np.argsort(
                    [v if np.isfinite(v) else np.inf for v in vals])]
                dom[est_name] = {
                    'values': {L: v for L, v in zip(labs, vals)},
                    'seed_mean': mean, 'seed_sd': sd,
                    'sigma': {L: ((v - mean) / sd
                                  if np.isfinite(v) and np.isfinite(sd)
                                  and sd > 0 else float('nan'))
                              for L, v in zip(labs, vals)},
                    'rank_order': order}
            print('  ranking by median: %s'
                  % ' < '.join(dom['median']['rank_order']))
            print('  winsor cap (q%g of all models pooled) = %.2f %s'
                  % (100 * WINSOR_QUANTILE, cap * scale, unit))
            print('  * trimmed_mean is NOT a fair cross-model comparator when '
                  'tail shapes differ')
            print('    (it removes a fixed FRACTION, so it removes more from '
                  'heavier tails). Use')
            print('    median or winsor_mean; see the note by TRIM_FRACTION '
                  'in this script.')
            out['metrics'][key]['domains'][dk] = dom

    # --- stability across models -------------------------------------------
    print('=' * 72)
    print('STABILITY -- structures with a per-structure force MAE above a '
          'threshold')
    out['stability'] = {}
    for dk in _DOMAIN_ORDER:
        if dk not in base_masks:
            continue
        m = base_masks[dk]
        print('-' * 72)
        print('  %s  (n=%d)' % (_DOMAIN_LABEL[dk], int(m.sum())))
        head = '  %-18s' % 'threshold eV/A'
        for L in labs:
            head += ' %10s' % L[:10]
        head += ' %12s %8s' % ('seed mean', 'seed sd')
        print(head)
        ent = {}
        for t in STABILITY_FORCE_THRESHOLDS:
            counts = [int((_worst_fill(a['f_mae_comp'][m]) > t).sum())
                      for a in aligned]
            mean, sd, _ = _seed_stats([float(c) for c in counts], labs)
            line = '  %-18s' % ('>%g' % t)
            for c in counts:
                line += ' %10d' % c
            line += ' %12.1f %8.1f' % (mean, sd)
            print(line)
            ent['>%g' % t] = {'counts': {L: c for L, c in zip(labs, counts)},
                              'seed_mean': mean, 'seed_sd': sd}
        out['stability'][dk] = ent

    # --- error concentration across models ---------------------------------
    print('=' * 72)
    print('ERROR CONCENTRATION -- structures carrying 50/90/99% of the total '
          'squared force error')
    print('  Quote this next to any pooled RMSE you print.')
    out['concentration'] = {}
    for dk in _DOMAIN_ORDER:
        if dk not in base_masks:
            continue
        m = base_masks[dk]
        print('-' * 72)
        print('  %s  (n=%d)' % (_DOMAIN_LABEL[dk], int(m.sum())))
        print('  %-18s %10s %10s %10s' % ('model', '50%', '90%', '99%'))
        ent = {}
        for L, a in zip(labs, aligned):
            sq = _worst_fill(a['f_sumsq'][m])
            fin = np.isfinite(sq)
            tot = float(sq[fin].sum())
            if tot <= 0:
                ks = [-1, -1, -1]
            else:
                cum = np.cumsum(np.sort(sq[fin])[::-1]) / tot
                ks = [int(np.searchsorted(cum, L2) + 1)
                      for L2 in (0.5, 0.9, 0.99)]
            print('  %-18s %10d %10d %10d' % (L, ks[0], ks[1], ks[2]))
            ent[L] = {'k50': ks[0], 'k90': ks[1], 'k99': ks[2],
                      'n_nonfinite': int((~fin).sum())}
        out['concentration'][dk] = ent

    # --- ceiling-relative across models ------------------------------------
    if ceiling is not None:
        print('=' * 72)
        print('RELATIVE TO THE FULL-DATA CEILING (%s)' % ceiling['__label__'])
        print('  median over structures of (model error - ceiling error), '
              'paired.')
        print('  This is the recommended headline metric: it answers "which')
        print('  subset comes closest to the model trained on everything?"')
        print('  with no threshold and no structures discarded.')
        out['ceiling'] = {'label': str(ceiling['__label__']), 'domains': {}}
        cel = _worst_fill(ceiling['f_mae_comp'])
        for dk in _DOMAIN_ORDER:
            if dk not in base_masks:
                continue
            m = base_masks[dk]
            print('-' * 72)
            print('  %s  (n=%d)' % (_DOMAIN_LABEL[dk], int(m.sum())))
            head = '  %-18s' % 'statistic'
            for L in labs:
                head += ' %10s' % L[:10]
            head += ' %12s %8s' % ('seed mean', 'seed sd')
            for L in labs:
                if not _is_seed_label(L):
                    head += ' %10s' % ('sd(%s)' % L[:5])
            print(head)
            ent = {}
            _ds = []
            for a in aligned:
                mine = _worst_fill(a['f_mae_comp'][m])
                c = cel[m]
                w = a['natoms'][m].astype(np.float64)
                d = mine - c
                fin = np.isfinite(d)
                good = np.isfinite(mine) & np.isfinite(c) & (c > 0)
                _ds.append((d[fin], w[fin], mine[good] / c[good], w[good]))
            ecap = _pooled_cap([x[0] for x in _ds], [x[1] for x in _ds])
            efloor = -ecap
            for stat, fn in (
                    ('excess_median',
                     lambda d, w, r, rw: _wquantile(d, w, 0.5)),
                    ('excess_winsor_mean',
                     lambda d, w, r, rw: _wwinsor_mean(d, w, ecap, efloor)),
                    ('excess_trimmed_mean*',
                     lambda d, w, r, rw: _wtrimmed_mean(d, w, TRIM_FRACTION)),
                    ('ratio_geomean',
                     lambda d, w, r, rw: _wgeomean(r, rw))):
                vals = []
                for d, w, r, rw in _ds:
                    vals.append(fn(d, w, r, rw))
                mean, sd, _ = _seed_stats(vals, labs)
                sc = 1.0 if stat == 'ratio_geomean' else 1000.0
                line = '  %-18s' % stat
                for v in vals:
                    line += ' %10.3f' % (v * sc)
                line += ' %12.3f %8.3f' % (mean * sc, sd * sc)
                for L, v in zip(labs, vals):
                    if not _is_seed_label(L):
                        line += ' %10s' % _fmt_sigma(v, mean, sd)
                print(line)
                order = [labs[i] for i in np.argsort(
                    [v if np.isfinite(v) else np.inf for v in vals])]
                ent[stat] = {'values': {L: v for L, v in zip(labs, vals)},
                             'seed_mean': mean, 'seed_sd': sd,
                             'sigma': {L: ((v - mean) / sd
                                           if np.isfinite(v)
                                           and np.isfinite(sd) and sd > 0
                                           else float('nan'))
                                       for L, v in zip(labs, vals)},
                             'rank_order': order}
            print('  ranking by excess_median: %s'
                  % ' < '.join(ent['excess_median']['rank_order']))
            print('  winsor cap +-%.2f meV/A (q%g of all models pooled)'
                  % (1000 * ecap, 100 * WINSOR_QUANTILE))
            print('  * excess_trimmed_mean is NOT comparable across models '
                  'here: a fixed-fraction')
            print('    trim removes more from heavier tails. Quote '
                  'excess_median (and excess_winsor_mean).')
            out['ceiling']['domains'][dk] = ent

        # ceiling-conditional profile, all models side by side
        print('-' * 72)
        print('  CEILING-CONDITIONAL PROFILE -- median force MAE per bin of '
              'the')
        print('  ceiling\'s own error. All bins shown. Compare candidates to')
        print('  each other within a bin; the ceiling column is biased inside')
        print('  its own bins (regression to the mean) and is shown only to')
        print('  locate the regime.')
        out['ceiling']['profile'] = {}
        edges = list(CEILING_BINS)
        for dk in _DOMAIN_ORDER:
            if dk not in base_masks:
                continue
            m = base_masks[dk]
            print('  %s' % _DOMAIN_LABEL[dk])
            head = '    %-14s %8s %9s %9s' % ('ceiling bin', 'n', '% of dom',
                                              'ceil med')
            for L in labs:
                head += ' %10s' % L[:10]
            head += ' %10s %8s' % ('seed mean', 'seed sd')
            for L in labs:
                if not _is_seed_label(L):
                    head += ' %9s' % ('sd(%s)' % L[:5])
            head += ' %8s' % 'best'
            print(head)
            prof = []
            for lo, hi in zip(edges[:-1], edges[1:]) :
                prof.append((('%g-%g' % (1000 * lo, 1000 * hi)),
                             m & (cel >= lo) & (cel < hi)))
            prof.append(('>=%g' % (1000 * edges[-1]),
                         m & (cel >= edges[-1])))
            rows_out = []
            n_skipped = 0
            for lab, sel in prof:
                n = int(sel.sum())
                if n < 20:
                    n_skipped += int(n > 0)
                    continue
                w = aligned[0]['natoms'][sel].astype(np.float64)
                cm = _wquantile(cel[sel], w, 0.5)
                vals = [_wquantile(_worst_fill(a['f_mae_comp'][sel]),
                                   a['natoms'][sel].astype(np.float64), 0.5)
                        for a in aligned]
                mean, sd, _ = _seed_stats(vals, labs)
                best = labs[int(np.argmin([v if np.isfinite(v) else np.inf
                                           for v in vals]))]
                line = '    %-14s %8d %8.2f%% %9.1f' % (
                    lab, n, 100.0 * n / max(int(m.sum()), 1), 1000 * cm)
                for v in vals:
                    line += ' %10.1f' % (v * 1000)
                line += ' %10.2f %8.2f' % (mean * 1000, sd * 1000)
                for L, v in zip(labs, vals):
                    if not _is_seed_label(L):
                        line += ' %9s' % _fmt_sigma(v, mean, sd)
                line += ' %8s' % best[:8]
                print(line)
                rows_out.append({'bin': lab, 'n': n, 'ceiling_median': cm,
                                 'values': {L: v for L, v in zip(labs, vals)},
                                 'seed_mean': mean, 'seed_sd': sd,
                                 'best': best})
            n_dom = int(m.sum())
            n_shown = sum(r['n'] for r in rows_out)
            if not rows_out:
                print('    (no bin with at least 20 structures)')
            elif n_skipped:
                print('    (%d structures in %d bin(s) with fewer than 20 '
                      'members, omitted from the rows above)'
                      % (n_dom - n_shown, n_skipped))
            # Coverage: the share of the DOMAIN, not of the printed bins. The
            # omitted structures still exist, so they belong in the
            # denominator; using the printed total instead inflates every
            # coverage figure slightly.
            if rows_out:
                cov = {}
                for r in rows_out:
                    cov[r['best']] = cov.get(r['best'], 0) + r['n']
                parts = ['%s %.2f%%' % (L, 100.0 * cov[L] / n_dom)
                         for L in labs if cov.get(L)]
                print('    bins where each model is best cover: %s'
                      % ', '.join(parts))
            out['ceiling']['profile'][dk] = rows_out

        # v2.1 ADDITION, guarded. Everything above this point has been
        # printed but NOTHING has been written to disk yet -- the save
        # block is below. An exception escaping here would therefore cost
        # the user paper_tables.json and every CSV, which is a far worse
        # outcome than losing the two extra tables. So this block may
        # fail, but it may not take the run down with it.
        try:
            # ------------------------------------------------------------------
            # v2.1 ADDITION: the same profile, scored per ATOM.
            # Two further passes over the same bins. Nothing above is touched; if
            # any model lacks per_atom_forces.npz the whole block prints one line
            # and is skipped.
            # ------------------------------------------------------------------
            # One entry per file to read: (label, path, matched order, the
            # per-structure table it must agree with).
            pa_jobs = [(L, q, it, t) for L, q, it, t
                       in zip(labs, paths, row_order, tabs)]
            pa_jobs.append(('__ceiling__', ceiling_path, ceiling_order,
                            ceiling_tab))
            pa_missing = [L for L, q, _, _ in pa_jobs
                          if not has_per_atom_forces(q)]
            print('-' * 72)
            if pa_missing:
                print('  PER-ATOM CEILING PROFILE -- skipped: no '
                      'per_atom_forces.npz for %s.'
                      % ', '.join('the ceiling' if L == '__ceiling__' else L
                                  for L in pa_missing))
                print('  Re-run those evaluations without --no_per_atom_forces.')
            else:
                # Bin masks are built ONCE, in base order, and reused for every
                # model: a per-model rebuild is where a paired comparison would
                # silently drift onto different structures.
                specs = {}
                for dk in _DOMAIN_ORDER:
                    if dk not in base_masks:
                        continue
                    specs[dk] = [(lab, sel) for lab, sel
                                 in _ceiling_bin_masks(cel, base_masks[dk])]
                # One model in memory at a time: 7 x ~78 MB at once is avoidable.
                stats = {}
                for L, q, it, t in pa_jobs:
                    view = _peratom_view(load_per_atom_forces(q, rows=t),
                                         own_order=it)
                    stats[L] = {dk: [_peratom_bin_stats(view, sel)
                                     for _, sel in specs[dk]] for dk in specs}
                    del view

                for field, tag, title in (
                        ('median_norm', 'profile_peratom_norm',
                         'median per-atom error ||dF_i|| over the bin\'s atoms'),
                        ('mae_f', 'profile_peratom_comp',
                         "MACE MAE_F over the bin's atoms (mean |dF_i,k| over "
                         'all 3N components)')):
                    print('-' * 72)
                    print('  CEILING-CONDITIONAL PROFILE, PER ATOM -- %s.' % title)
                    print('  Same structures and same bins as the table above, so '
                          'the rows line up;')
                    print('  the statistic is taken over the bin\'s ATOMS with no '
                          'per-structure mean')
                    print('  taken first. The ceiling column labels the regime, '
                          'not its skill.')
                    out['ceiling'][tag] = {}
                    for dk in specs:
                        print('  %s' % _DOMAIN_LABEL[dk])
                        head = ('    %-14s %8s %11s %9s %9s'
                                % ('ceiling bin', 'n', 'n atoms', '% of dom',
                                   'ceil'))
                        for L in labs:
                            head += ' %10s' % L[:10]
                        head += ' %10s %8s' % ('seed mean', 'seed sd')
                        for L in labs:
                            if not _is_seed_label(L):
                                head += ' %9s' % ('sd(%s)' % L[:5])
                        head += ' %8s' % 'best'
                        print(head)
                        m = base_masks[dk]
                        rows_pa = []
                        n_skipped = 0
                        for bi, (blab, sel) in enumerate(specs[dk]):
                            n = int(sel.sum())
                            if n < 20:
                                n_skipped += int(n > 0)
                                continue
                            na = stats[labs[0]][dk][bi]['n_atoms']
                            cv = stats['__ceiling__'][dk][bi][field]
                            vals = [stats[L][dk][bi][field] for L in labs]
                            mean, sd, _ = _seed_stats(vals, labs)
                            best = labs[int(np.argmin(
                                [v if np.isfinite(v) else np.inf for v in vals]))]
                            line = ('    %-14s %8d %11d %8.2f%% %9.1f'
                                    % (blab, n, na,
                                       100.0 * n / max(int(m.sum()), 1),
                                       1000 * cv))
                            for v in vals:
                                line += ' %10.1f' % (v * 1000)
                            line += ' %10.2f %8.2f' % (mean * 1000, sd * 1000)
                            for L, v in zip(labs, vals):
                                if not _is_seed_label(L):
                                    line += ' %9s' % _fmt_sigma(v, mean, sd)
                            line += ' %8s' % best[:8]
                            print(line)
                            rows_pa.append({'bin': blab, 'n': n, 'n_atoms': na,
                                            'ceiling_value': cv,
                                            'values': {L: v for L, v
                                                       in zip(labs, vals)},
                                            'seed_mean': mean, 'seed_sd': sd,
                                            'best': best})
                        if not rows_pa:
                            print('    (no bin with at least 20 structures)')
                        elif n_skipped:
                            n_shown = sum(r['n'] for r in rows_pa)
                            print('    (%d structures in %d bin(s) with fewer '
                                  'than 20 members, omitted from the rows above)'
                                  % (int(m.sum()) - n_shown, n_skipped))
                        out['ceiling'][tag][dk] = rows_pa
        except Exception as exc:                     # noqa: BLE001
            # Any partially written entry is dropped, so the JSON never
            # carries half a table.
            for _tag in ('profile_peratom_norm', 'profile_peratom_comp'):
                out['ceiling'].pop(_tag, None)
            print('-' * 72)
            print('  PER-ATOM CEILING PROFILE -- skipped after an error:')
            print('    %s: %s' % (type(exc).__name__, exc))
            print('  Every other table in this report is unaffected and '
                  'is still written below.')


        # v4 ADDITION, guarded exactly like the block above and for the same
        # reason: these two tables may fail, but they may not take the run
        # down with them. Everything above has been printed; nothing has been
        # written to disk yet.
        try:
            pa_jobs4 = [(L, q, it, t) for L, q, it, t
                        in zip(labs, paths, row_order, tabs)]
            ceil_job4 = ('__ceiling__', ceiling_path, ceiling_order,
                         ceiling_tab)
            miss4 = [L for L, q, _o, _t in pa_jobs4 + [ceil_job4]
                     if not has_per_atom_forces(q)]
            print('-' * 72)
            if miss4:
                print('  PER-ATOM-BINNED CEILING PROFILES -- skipped: no '
                      'per_atom_forces.npz for %s.'
                      % ', '.join('the ceiling' if L == '__ceiling__' else L
                                  for L in miss4))
                print('  Re-run those evaluations without '
                      '--no_per_atom_forces.')
            else:
                # The common atom order is the first model's, taken in the
                # common ROW order. natoms is not used for it: a structure
                # that carried no forces contributes no atoms, and the
                # per-atom file is the only thing that knows.
                with np.load(_per_atom_forces_path(paths[0]),
                             allow_pickle=False) as _z0:
                    _counts0 = np.diff(_z0['struct_offset'].astype(np.int64))
                base_counts4 = _counts0[np.asarray(row_order[0],
                                                   dtype=np.int64)]
                ab = _atombin_profiles(labs, pa_jobs4, ceil_job4,
                                       base_counts4, base_masks)
                _print_atombin(
                    ab, labs, 'norm', 1000.0, '%10.1f', 'meV/A',
                    'CEILING-CONDITIONAL PROFILE, BINNED PER ATOM -- median '
                    '||dF_i||.',
                    ("ATOMS, not structures, are binned: a bin holds the "
                     "atoms whose own",
                     "ceiling error ||dF_i|| falls between the edges, and the "
                     "statistic is the",
                     "median of ||dF_i|| over exactly those atoms. The norm "
                     "convention is used",
                     "throughout, NOT MACE's component-average MAE_F. The "
                     "ceiling column is",
                     "biased low inside its own bins, and more strongly than "
                     "in the",
                     "per-structure profile above, because one atom's error "
                     "is far noisier",
                     "than a structure average; read it as the regime, never "
                     "as skill."))
                _print_atombin(
                    ab, labs, 'rel', 1.0, '%10.4f', '-',
                    'CEILING-CONDITIONAL PROFILE, BINNED PER ATOM -- median '
                    'RELATIVE error.',
                    ("rho_i = ||dF_i|| / ||F_ref,i||, dimensionless. Bins "
                     "hold the atoms whose",
                     "ceiling rho_i falls between the edges. rho = 1 is an "
                     "edge on purpose: every",
                     "bin above it is 'the error is at least as large as the "
                     "force it should",
                     "reproduce'. The denominator is the DFT reference and is "
                     "therefore the same",
                     "for every model, so this is a REWEIGHTING of the same "
                     "per-atom errors, not",
                     "a new measurement; which model wins on any single atom "
                     "cannot change.",
                     "The ceiling column is biased low inside its own bins."))
                out['ceiling']['profile_atombin_norm'] = ab['norm']
                out['ceiling']['profile_atombin_rel'] = ab['rel']
                out['ceiling']['atombin_info'] = ab['info']
        except Exception as exc:                     # noqa: BLE001
            for _tag in ('profile_atombin_norm', 'profile_atombin_rel',
                         'atombin_info'):
                out['ceiling'].pop(_tag, None)
            print('-' * 72)
            print('  PER-ATOM-BINNED CEILING PROFILES -- skipped after an '
                  'error:')
            print('    %s: %s' % (type(exc).__name__, exc))
            print('  Every other table in this report is unaffected and '
                  'is still written below.')
    # v5 ADDITION, guarded exactly like the two per-atom blocks above and for
    # the same reason: this table may fail, but it may not take the run down
    # with it. Everything above has been printed; nothing has been written to
    # disk yet. It sits OUTSIDE the `if ceiling is not None` block because it
    # is not a ceiling-relative table -- it needs no ceiling and is printed
    # with or without one. When there is a ceiling it is carried as one extra
    # column, as the combined CDF file already does, and it never enters the
    # seed mean +- sd.
    try:
        blow_jobs = [(L, q, it, t) for L, q, it, t
                     in zip(labs, paths, row_order, tabs)]
        blow_cols = list(labs)
        blow_names = list(labs)
        if ceiling is not None:
            blow_jobs.append(('__ceiling__', ceiling_path, ceiling_order,
                              ceiling_tab))
            blow_cols.append('__ceiling__')
            blow_names.append(str(ceiling['__label__']))
        miss5 = [L for L, q, _o, _t in blow_jobs
                 if not has_per_atom_forces(q)]
        if miss5:
            print('=' * 72)
            print('BLOW-UP CENSUS -- skipped: no per_atom_forces.npz for %s.'
                  % ', '.join('the ceiling' if L == '__ceiling__' else L
                              for L in miss5))
            print('  Re-run those evaluations without --no_per_atom_forces.')
        else:
            blowup = _tbl_peratom_blowup(labs, blow_jobs,
                                         _blowup_domain_masks(base_masks))
            _print_peratom_blowup(blowup, labs, blow_cols, blow_names)
            blowup['column_names'] = {L: n for L, n
                                      in zip(blow_cols, blow_names)}
            out['blowup_atoms'] = blowup
    except Exception as exc:                         # noqa: BLE001
        # Any partially written entry is dropped, so the JSON never carries
        # half a table.
        out.pop('blowup_atoms', None)
        print('-' * 72)
        print('  BLOW-UP CENSUS -- skipped after an error:')
        print('    %s: %s' % (type(exc).__name__, exc))
        print('  Every other table in this report is unaffected and is still '
              'written below.')

    # v5 ADDITION, guarded like every block above it. Independent of the
    # census: it shares only the domain list, so that the two new tables are
    # reported over exactly the same populations. Unlike every other per-atom
    # table here it needs no per_atom_forces.npz -- those are used only to
    # CHECK it -- so it survives a run where the per-atom files are missing.
    try:
        mae_jobs = list(zip(labs, aligned))
        mae_cols = list(labs)
        mae_names = list(labs)
        if ceiling is not None:
            mae_jobs.append(('__ceiling__', ceiling))
            mae_cols.append('__ceiling__')
            mae_names.append(str(ceiling['__label__']))
        mae_domains = _blowup_domain_masks(base_masks)
        pamae = _peratom_force_mae_table(labs, mae_jobs, mae_domains)

        # Guard 1's input: Table 1's own pooled row, if that metric ran.
        pooled_row = {}
        for dk, blk in (out['metrics'].get('f_mae_comp', {})
                        .get('domains', {}) or {}).items():
            if 'pooled_mean_MAE' in blk:
                pooled_row[dk] = blk['pooled_mean_MAE']['values']

        # Guard 2's input: the same numbers from the raw atoms, when readable.
        direct = None
        pa_jobs5 = [(L, q, it, t) for L, q, it, t
                    in zip(labs, paths, row_order, tabs)]
        if ceiling is not None:
            pa_jobs5.append(('__ceiling__', ceiling_path, ceiling_order,
                             ceiling_tab))
        if all(has_per_atom_forces(q) for _L, q, _o, _t in pa_jobs5):
            try:
                direct = _peratom_force_mae_direct(pa_jobs5, mae_domains)
            except Exception as exc2:            # noqa: BLE001
                direct = None
                print('-' * 72)
                print('  (per-atom cross-check of the force MAE unavailable: '
                      '%s: %s)' % (type(exc2).__name__, exc2))

        _print_peratom_force_mae(pamae, labs, mae_cols, mae_names,
                                 direct=direct, pooled_row=pooled_row)
        pamae['column_names'] = {L: n for L, n in zip(mae_cols, mae_names)}
        out['peratom_force_mae'] = pamae
    except Exception as exc:                     # noqa: BLE001
        out.pop('peratom_force_mae', None)
        print('-' * 72)
        print('  PER-ATOM FORCE MAE -- skipped after an error:')
        print('    %s: %s' % (type(exc).__name__, exc))
        print('  Every other table in this report is unaffected and is still '
              'written below.')

    # --- save ---------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'paper_tables.json'), 'w') as fh:
        json.dump(_jsonable(out), fh, indent=1, sort_keys=True)

    # tidy CSVs, one per (metric, domain), models as columns
    for key, blk in out['metrics'].items():
        for dk, dom in blk['domains'].items():
            fn = os.path.join(out_dir, 'table_%s_%s.csv' % (key, dk))
            with open(fn, 'w') as fh:
                fh.write('estimator,' + ','.join(labs) +
                         ',seed_mean,seed_sd,' +
                         ','.join('%s_sigma' % L for L in labs) + '\n')
                for est, e in dom.items():
                    fh.write(est + ',' +
                             ','.join('%.8g' % e['values'][L] for L in labs) +
                             ',%.8g,%.8g,' % (e['seed_mean'], e['seed_sd']) +
                             ','.join('%.4g' % e['sigma'][L]
                                      for L in labs) + '\n')

    # v2.1 ADDITION: the per-atom ceiling profiles as tidy CSVs.
    for tag, stat in (('profile_peratom_norm', 'median ||dF_i|| [meV/A]'),
                      ('profile_peratom_comp', 'MACE MAE_F [meV/A]')):
        for dk, rws in (out.get('ceiling', {}).get(tag, {}) or {}).items():
            fn = os.path.join(out_dir, 'ceiling_%s_%s.csv' % (tag, dk))
            with open(fn, 'w') as fh:
                fh.write('# %s, over the atoms of each ceiling bin\n' % stat)
                fh.write('bin_meV_per_A,n_structures,n_atoms,ceiling,'
                         + ','.join(labs) + ',seed_mean,seed_sd,best\n')
                for r in rws:
                    fh.write('%s,%d,%d,%.8g,' % (r['bin'], r['n'],
                                                 r['n_atoms'],
                                                 1000 * r['ceiling_value']))
                    fh.write(','.join('%.8g' % (1000 * r['values'][L])
                                      for L in labs))
                    fh.write(',%.8g,%.8g,%s\n' % (1000 * r['seed_mean'],
                                                   1000 * r['seed_sd'],
                                                   r['best']))


    # v4 ADDITION: the per-atom-binned profiles as tidy CSVs.
    for tag, stat, sc in (
            ('profile_atombin_norm', 'median ||dF_i|| [meV/A]', 1000.0),
            ('profile_atombin_rel',
             'median rho_i = ||dF_i||/||F_ref,i|| [dimensionless]', 1.0)):
        for dk, rws in (out.get('ceiling', {}).get(tag, {}) or {}).items():
            fn = os.path.join(out_dir, 'ceiling_%s_%s.csv' % (tag, dk))
            with open(fn, 'w') as fh:
                fh.write('# %s, over the ATOMS of each ceiling bin\n' % stat)
                fh.write('# bins are cut on the ceiling model\'s own per-atom '
                         'value, not on a per-structure mean\n')
                if tag == 'profile_atombin_rel':
                    fh.write('# atoms with ||F_ref,i|| <= %g eV/A excluded\n'
                             % REL_MIN_REF_FORCE)
                fh.write('bin,n_atoms,pct_of_domain,ceiling,'
                         + ','.join(labs) + ',seed_mean,seed_sd,best\n')
                for r in rws:
                    fh.write('%s,%d,%.6g,%.8g,'
                             % (r['bin'], r['n'], r['pct'],
                                sc * r['ceiling_value']))
                    fh.write(','.join('%.8g' % (sc * r['values'][L])
                                      for L in labs))
                    fh.write(',%.8g,%.8g,%s\n' % (sc * r['seed_mean'],
                                                  sc * r['seed_sd'],
                                                  r['best']))
    # combined CDF for the figure: one file, models as columns
    for dk in _DOMAIN_ORDER:
        if dk not in base_masks:
            continue
        m = base_masks[dk]
        fn = os.path.join(out_dir, 'cdf_all_models_%s.csv' % dk)
        with open(fn, 'w') as fh:
            fh.write('threshold_meV_per_A,' + ','.join(labs) +
                     (',%s' % ceiling['__label__'] if ceiling else '') + '\n')
            for t in CDF_FORCE_GRID:
                cells = []
                for a in aligned:
                    v = _worst_fill(a['f_mae_comp'][m])
                    w = a['natoms'][m].astype(np.float64)
                    cells.append(float(w[v <= t].sum() / w.sum()))
                if ceiling:
                    v = cel[m]
                    w = aligned[0]['natoms'][m].astype(np.float64)
                    cells.append(float(w[v <= t].sum() / w.sum()))
                fh.write('%.6g,' % (1000 * t) +
                         ','.join('%.8g' % c for c in cells) + '\n')

    # v5 ADDITION: the blow-up census as tidy CSVs, one per domain, models as
    # columns. Counts first, then the same numbers as a percentage of that
    # model's own atom count in the domain, in '<label>_pct' columns -- the
    # layout the table_*.csv files already use for their '<label>_sigma'
    # columns. Thresholds are in eV/A, NOT meV/A as the CDF file's grid is:
    # this is a census, not a curve, and a blow-up threshold is quoted in eV/A
    # everywhere else in this script.
    blow = out.get('blowup_atoms')
    if blow:
        cnames = blow.get('column_names', {})
        bcols = list(blow['columns'])
        hdr = [cnames.get(L, L) for L in bcols]
        for dk, ent in blow['domains'].items():
            fn = os.path.join(out_dir, 'blowup_atoms_%s.csv' % dk)
            with open(fn, 'w') as fh:
                fh.write('# ATOMS whose own force error norm ||dF_i|| exceeds '
                         'the threshold, domain %s\n' % dk)
                fh.write('# strict test (> threshold); norm convention, NOT '
                         "MACE's component-average MAE_F,\n")
                fh.write('# and no per-structure mean taken first\n')
                fh.write('# n_atoms in domain: %s\n'
                         % ', '.join('%s %d' % (h, ent['n_atoms'][L])
                                     for h, L in zip(hdr, bcols)))
                fh.write('# non-finite ||dF_i||, counted above every '
                         'threshold: %s\n'
                         % ', '.join('%s %d' % (h, ent['n_nonfinite'][L])
                                     for h, L in zip(hdr, bcols)))
                fh.write('threshold_eV_per_A,' + ','.join(hdr) +
                         ',seed_mean,seed_sd,' +
                         ','.join('%s_pct' % h for h in hdr) + '\n')
                for t in blow['thresholds']:
                    e = ent['thresholds']['>%g' % t]
                    fh.write('%.6g,' % t +
                             ','.join('%d' % e['counts'][L] for L in bcols) +
                             ',%.8g,%.8g,' % (e['seed_mean'], e['seed_sd']) +
                             ','.join('%.6g' % (100.0 * e['fractions'][L])
                                      for L in bcols) + '\n')

    # v5 ADDITION: the per-atom force MAE as tidy CSVs, one per domain, the
    # two conventions as rows and the models as columns. Values are written in
    # meV/A and the header says so: the table_*.csv files above write their
    # values UNSCALED (eV/A) with no unit anywhere in the file, which is a
    # trap worth not repeating in a new file.
    pam = out.get('peratom_force_mae')
    if pam:
        pnames = pam.get('column_names', {})
        pcols = list(pam['columns'])
        phdr = [pnames.get(L, L) for L in pcols]
        for dk, ent in pam['domains'].items():
            fn = os.path.join(out_dir, 'peratom_force_mae_%s.csv' % dk)
            with open(fn, 'w') as fh:
                fh.write('# pooled PER-ATOM force MAE over domain %s, '
                         'in meV/A\n' % dk)
                fh.write('# norm = %s\n' % pam['conventions']['norm'])
                fh.write('# comp = %s\n' % pam['conventions']['comp'])
                fh.write('# the two are different measurements of the same '
                         'errors; their ratio is bounded by [1, 3]\n')
                fh.write('# and equals 2 only for isotropic errors, so they '
                         'may never be compared with each other\n')
                fh.write('# %d structures, %d atoms (sum of natoms over the '
                         'domain)\n' % (ent['n_structures'], ent['n_atoms']))
                fh.write('convention,' + ','.join(phdr) +
                         ',seed_mean,seed_sd,' +
                         ','.join('%s_sigma' % h for h in phdr) + '\n')
                for tag, _key, _desc in _PERATOM_MAE_KEYS:
                    e = ent[tag]
                    fh.write(tag + ',' +
                             ','.join('%.8g' % (1000 * e['values'][L])
                                      for L in pcols) +
                             ',%.8g,%.8g,' % (1000 * e['seed_mean'],
                                              1000 * e['seed_sd']) +
                             ','.join('%.4g' % e['sigma'][L]
                                      for L in pcols) + '\n')

    print('=' * 72)
    print('Aggregated tables written to %s/' % out_dir)
    for f in sorted(os.listdir(out_dir)):
        print('  %s' % f)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate a MACE model on an extxyz dataset and print the '
                    'energy, force and stress MAE/RMSE.')
    p.add_argument('--configs', default=TEST_SET_PATH,
                   help='reference extxyz dataset')
    p.add_argument('--model', default=MODEL_PATH, help='MACE model file')
    p.add_argument('--output', default=OUT_PATH,
                   help='extxyz file the predictions are written to')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--default_dtype', default='float32',
                   choices=['float32', 'float64'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--info_prefix', default='MACE_',
                   help='prefix eval_configs used for the predicted keys')
    p.add_argument('--no_compute_stress', action='store_true',
                   help='do not ask eval_configs for stress (use for a model '
                        'that cannot produce it); the stress section of the '
                        'report is then empty')
    p.add_argument('--skip_eval', action='store_true',
                   help='do not run the model; only report errors from an '
                        'existing --output file. Note that an output file '
                        'produced without --compute_stress carries no stress.')
    p.add_argument('--force_error_cutoffs', type=float, nargs='*',
                   default=None, metavar='EV_PER_A',
                   help='force-error thresholds in eV/A; the metrics are '
                        'reported again with the atoms above each threshold '
                        'excluded. Pass with no values for the uncut report '
                        f'only. Default: {FORCE_ERROR_CUTOFFS}')
    p.add_argument('--force_error_cutoff_on', default=None,
                   choices=sorted(CUTOFF_CRITERIA),
                   help='per-atom statistic the cutoffs apply to: '
                        "'max_component' = component-wise, max_k |dF_i,k| "
                        "(MACE's RMSE_F quantity); 'norm' = ||dF_i||. "
                        f'Default: {FORCE_ERROR_CUTOFF_ON}')
    p.add_argument('--non_bulk_detector', default='__default__',
                   choices=sorted(NON_BULK_CRITERIA) + ['none'],
                   help='report the metrics split into a bulk and a non-bulk '
                        'half. "none" disables the split. Everything except '
                        f'"packing" builds a neighbour list. Default: '
                        f'{NON_BULK_DETECTOR}')
    p.add_argument('--non_bulk_packing_threshold', type=float, default=None,
                   help='packing fraction below which a cell counts as '
                        f'non-bulk. Default: {NON_BULK_PACKING_THRESHOLD}')
    p.add_argument('--contact_ratio_cutoff', default='__default__',
                   help='report structures whose smallest pair distance falls '
                        'below this multiple of the sum of the two covalent '
                        'radii as their own "overcompressed" domain. Pass '
                        '"none" to disable. Default: '
                        f'{CONTACT_RATIO_CUTOFF}')
    # --- v2 -------------------------------------------------------------
    p.add_argument('--paper_dir', default=None,
                   help='directory the per-structure table, the JSON tables '
                        'and the plot CSVs are written to. Default: '
                        "'<output without extension>_paper'")
    p.add_argument('--label', default=None,
                   help='short name for this model in the saved tables (e.g. '
                        'v14, rdm1). Label the random-seed replicas rdm1..rdm4 '
                        'so --aggregate recognises them as the noise floor. '
                        'Default: the MODEL_LABEL constant, else derived from '
                        '--model.')
    p.add_argument('--no_per_structure', action='store_true',
                   help='do not collect the per-structure table; restores the '
                        'v1 memory profile and skips every paper section')
    p.add_argument('--save_per_atom', action='store_true',
                   help='also dump every per-atom force error (float32, '
                        '~50 MB for MatPES) for a per-atom distribution plot')
    p.add_argument('--no_per_atom_forces', action='store_true',
                   help='do NOT write per_atom_forces.npz (the raw, unbinned '
                        'per-atom force error, about 26 bytes per atom). It '
                        'is written by default; without it PAPER TABLES 6b '
                        'and 6c cannot be produced.')
    p.add_argument('--ceiling', default=None, metavar='PER_STRUCTURE_NPZ',
                   help='the CEILING MODEL\'s per_structure.npz, or the '
                        '--paper_dir containing it (e.g. MACE-OMAT-0, i.e. a '
                        'model trained on the full dataset). Its own error is '
                        'against the DFT reference and is not zero. '
                        'Enables the paired ceiling-relative tables.')
    p.add_argument('--aggregate', nargs='+', default=None,
                   metavar='DIR_OR_NPZ',
                   help='do not evaluate anything: read several per-model '
                        'results and print/save the combined paper tables '
                        'with the random-seed noise floor. Each argument is '
                        'either a --paper_dir from a previous run or the '
                        'per_structure.npz inside it, so per-model output can '
                        'stay next to each model. Names come from each run\'s '
                        'meta.json (falling back to the directory name), so '
                        '--labels is only needed to override them.')
    p.add_argument('--labels', nargs='+', default=None,
                   help='override the names of the --aggregate inputs, in the '
                        'same order. Default: each run\'s stored label, else '
                        'its directory name. Labels starting with rdm/seed/'
                        'random are treated as the random-seed replicas that '
                        'define the noise floor.')
    p.add_argument('--aggregate_non_bulk_detector', default=None,
                   choices=sorted(NON_BULK_CRITERIA),
                   help='v3, --aggregate only: recompute the bulk / non-bulk '
                        'split from the geometric descriptors already stored '
                        'in every per_structure.npz, instead of using the '
                        'classification saved at evaluation time. Applied '
                        'identically to every model AND to the --ceiling, so '
                        'the paired comparisons stay on one definition. The '
                        'overcompressed domain is left as stored. Only the '
                        'predicate changes: the descriptors were built at '
                        f'NON_BULK_BOND_SCALE={NON_BULK_BOND_SCALE:g} during '
                        'the evaluation and cannot be rebuilt here, so '
                        'changing the bond scale still needs a re-evaluation. '
                        'Default: use the stored classification, i.e. behave '
                        'exactly like v2.')
    p.add_argument('--aggregate_out', default='matpes_paper_tables',
                   help='output directory for --aggregate. '
                        'Default: matpes_paper_tables')
    # Subsampling is configured by the FRACTION / SEED / SUBSET_FILE /
    # KEEP_SUBSET constants at the top of this file, not on the command line.
    return p.parse_args()


def main():
    args = parse_args()

    if args.aggregate:
        aggregate_models(args.aggregate, labels=args.labels,
                         out_dir=args.aggregate_out,
                         ceiling_path=args.ceiling,
                         agg_non_bulk_detector=args.aggregate_non_bulk_detector)
        return

    # TEST_SET_PATH / MODEL_PATH are None by default (they used to be the
    # author's absolute paths), so fail here with a message naming the flag
    # rather than deep inside ASE or MACE with a confusing one.
    if not args.configs:
        raise SystemExit(
            '--configs is required: the MatPES test set as a single extxyz '
            'file (the published run used MatPES PBE 2025.2 with rare earths '
            'removed, the same file stage 06 uses as test_file). Pass '
            '--configs, or set TEST_SET_PATH in the USER CONFIGURATION block '
            'at the top of this file. See DATA.md.')
    if not args.model and not args.skip_eval:
        raise SystemExit(
            '--model is required: the *_compiled.model exported by stage 08. '
            'Pass --model, or set MODEL_PATH in the USER CONFIGURATION block '
            'at the top of this file. (--skip_eval reports on an existing '
            '--output file and needs no model.)')

    configs = args.configs
    sample_info = None
    subset_path = None

    if FRACTION != 1.0:
        offsets = build_frame_offsets(args.configs)
        indices = select_frame_indices(len(offsets), FRACTION, SEED)
        subset_path = SUBSET_FILE or (
            os.path.splitext(args.output)[0] + '_subset.xyz')
        write_frame_subset(args.configs, offsets, indices, subset_path)
        configs = subset_path
        sample_info = {'fraction': FRACTION, 'seed': SEED,
                       'n_selected': len(indices), 'n_total': len(offsets)}
        print(f'Sampled {len(indices)} of {len(offsets)} frames '
              f'(FRACTION={FRACTION:g}, SEED={SEED}) -> {subset_path}',
              flush=True)

    if args.skip_eval:
        print(f'Skipping evaluation, reporting on existing {args.output}',
              flush=True)
    else:
        eval_mace(configs=configs, model=args.model, output=args.output,
                  device=args.device, default_dtype=args.default_dtype,
                  batch_size=args.batch_size,
                  compute_stress=not args.no_compute_stress)
        print(f'Predictions written to {args.output}', flush=True)

    try:
        report_efs_errors(configs, args.output,
                          info_prefix=args.info_prefix,
                          sample_info=sample_info,
                          force_error_cutoffs=args.force_error_cutoffs,
                          cutoff_on=args.force_error_cutoff_on,
                          non_bulk_detector=(
                              None if args.non_bulk_detector == 'none'
                              else args.non_bulk_detector),
                          non_bulk_packing_threshold=(
                              args.non_bulk_packing_threshold),
                          contact_ratio_cutoff=(
                              None if args.contact_ratio_cutoff == 'none'
                              else args.contact_ratio_cutoff),
                          collect_per_structure=not args.no_per_structure,
                          paper_dir=(args.paper_dir or PAPER_OUTPUT_DIR
                                     or os.path.splitext(args.output)[0]
                                     + '_paper'),
                          ceiling_path=args.ceiling,
                          save_per_atom=args.save_per_atom or None,
                          save_per_atom_forces=(
                              False if args.no_per_atom_forces else None),
                          model_label=(args.label or MODEL_LABEL
                                       or _default_label(args.model)))
    finally:
        # The subset is reproducible from (SEED, FRACTION, configs), so it is
        # removed by default. Set KEEP_SUBSET = True to keep it, e.g. to
        # re-report with --skip_eval without regenerating it.
        if subset_path and not KEEP_SUBSET and os.path.exists(subset_path):
            os.remove(subset_path)


if __name__ == '__main__':
    main()
