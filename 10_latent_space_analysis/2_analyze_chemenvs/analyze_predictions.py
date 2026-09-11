#!/usr/bin/env python3


import ase.io
from ase.data import covalent_radii, atomic_numbers
import numpy as np

from pymatgen.core import Structure
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import Counter

from mpi4py import MPI

import time
import os
import shutil
from io import StringIO

# ============================================================================
# Tokenises the structures of stage 10.1 and attaches each site's force and
# energy error to its token.
#
# The token is
#     <element>_<spin>_<coord_elements>_<coord_symbol>_<perturb_scale>_<rmin_bin>_<a_bin>
#
# Two things about it catch people out:
#   * the LAST field is the asymmetry bin, not a distance bin;
#   * the field COUNT varies, because 11 chemenv polyhedron symbols contain an
#     underscore. Index the trailing fields from the END (a_bin = parts[-1],
#     rmin_bin = parts[-2]) or anchor on the ':' in the symbol.
#
# The constants block, _covalent_radius(), _distance_ratio_bins(),
# get_complete_coordination_info() and build_coord_descriptor() must behave
# exactly as their counterparts in stage 01's tokenize_omat24.py, otherwise the
# database-side and prediction-side tokens stop being the same join key. Keep
# the two in sync.
#
# The I/O below is built to stay fast and memory-bounded over millions of
# structures:
#
#   1. RANDOM-ACCESS READS via a byte-offset index instead of per-frame
#      ase.io.read(file, index=idx), which re-scans the whole text file on every
#      call and so costs O(N^2) over a dataset. Each rank builds a list of frame
#      byte offsets in one sequential pass and then seek()s straight to its own
#      frames, parsing each in O(1).
#
#   1. RANDOM-ACCESS READS via a byte-offset index instead of per-frame
#      ase.io.read(file, index=idx). The latter re-scans the whole text file on
#      EVERY call (cost grows with frame position -> O(N^2) over the dataset; we
#      measured ~2 s/frame on a 320k-frame file). Each rank instead builds a list
#      of frame byte offsets in a single sequential pass (~1 us/frame) and then
#      seek()s directly to only its own frames, parsing each in O(1)
#
#   2. INCREMENTAL PER-CHUNK WRITES. The original accumulated every processed
#      Atoms object in `local_atoms` until the very end, so peak memory grew with
#      Here each chunk is appended to the rank's output file and then dropped, so
#      per-rank memory is bounded by CHUNK_SIZE regardless of dataset size.
#
#   3. STREAMING SCALAR REDUCTION for the global MAE/RMSE. The original gathered
#      every per-atom error array onto rank 0 (a multi-GB spike at scale). Here
#      each rank keeps running sums (sum|e|, sum e^2, count) and we comm.reduce
#      six scalars at the end. The printed MAE/RMSE therefore match the original
#      to floating-point epsilon (summation order differs); the per-structure
#      output arrays are unchanged.
# ============================================================================

# REPO CHANGE: overridable via the environment; the defaults are the as-run
# values. The input default assumes stage 10.1 wrote one directory up.
MACE_PRED_EXTXYZ = os.environ.get(
    "CHEMENV_PRED_XYZ", "../all_structures_ref_and_predicted.xyz")
OUTPUT_XYZ = os.environ.get(
    "CHEMENV_OUT_XYZ", "all_structures_with_errors_and_v4chemenvs_mace_omat_medium.xyz")
BANNED_ELEMENTS = set([])
CHUNK_SIZE = 100  # structures per chunk per worker — tune this

lgf = None
dataset = None

def contains_rare_earths(structure):
    rare_earth_elements = {'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'}
    if bool(set(structure.get_chemical_symbols()).intersection(rare_earth_elements)):
        return True
    else:
        return False

def get_complete_coordination_info(lse, structure, site_index, strategy=None):
    neighbors_info = None
    if lse is not None:
        # Get coordination environment
        coord_env = lse.coordination_environments[site_index]
        neighbors_info = lse.neighbors_sets[site_index]

    if lse is not None and neighbors_info:           # truthy: handles both None and []
            result = {
                'central_atom': structure[site_index].specie.symbol,
                'coordination_environments': coord_env,
                'neighbor_elements': [n.specie.symbol for n in neighbors_info[0].neighb_sites],
                #The neighbour objects THEMSELVES, so that the distance-ratio fields are
                #computed from exactly the shell that produced 'neighbor_elements'.
                'neighbor_sites': list(neighbors_info[0].neighb_sites),
            }

    else:
        result = {
            'central_atom': structure[site_index].specie.symbol,
            'coordination_environments': 'INVALID',
            'neighbor_elements': 'INVALID',
            'neighbor_sites': []
            }

    return result


# ----------------------------------------------------------------------------
# Adaptive per-atom distance cutoff (two-cutoff selector)
#
# chemenv's SimplestChemenvStrategy selects neighbours with a distance cutoff
# that is RELATIVE to the nearest neighbour. The default (1.4) therefore
# under-coordinates atoms sitting in shells of very different bond lengths
# (e.g. a phosphate O bonded to a short P and a long Ca, or N2 trapped in a
# metal cage): once distances are normalised to the closest contact, the longer
# but genuine bonds fall past the cutoff and the atom collapses to S:1.
#
# We borrow VoronoiNN's solid-angle weight purely as a DISCOVERY signal to
# decide, per atom, how far chemenv should reach — while leaving chemenv as the
# geometry CLASSIFIER. For each representative site we build the neighbour set at
# the default cutoff (1.4) and at an extended cutoff (1.7) and adopt the extended
# set only if at least one neighbour it ADDS has a normalised Voronoi solid-angle
# weight >= VORONOI_PROMINENCE_THRESHOLD (i.e. that extra contact occupies a real
# fraction of the central atom's surroundings, not a screened sliver). Otherwise
# we keep the default 1.4 set. The choice is bounded to [1.4, 1.7]: it can never
# collapse below 1.4 (avoids CrystalNN-style under-coordination of rattled
# structures) nor extend past 1.7 (avoids running into the continuum of a dense
# metal). The selection is per ATOM and based only on composition/geometry, so it
# is applied identically to every dataset (MatPES, OMat24) with no data-origin
# branching.
#
# This block is the single source of truth for the selector and is byte-identical
# in stage 01's tokenize_omat24.py — keep the two in sync.
DISTANCE_CUTOFF_DEFAULT = 1.4
DISTANCE_CUTOFF_EXTENDED = 1.7
ANGLE_CUTOFF = 0.3
VORONOI_PROMINENCE_THRESHOLD = 0.45


# ----------------------------------------------------------------------------
# Distance-ratio field (rmin) and the asymmetry field A.
#
# The rest of the token is scale-invariant: chemenv rescales the coordination
# polyhedron before scoring it, so a structure and the same structure compressed
# by 20% give the same token. These two fields put the length scale back.
#
# For every contact i of the neighbour set we form a reduced length
#
#     r_i = d_i / (R_cov[central] + R_cov[neighbour_i])
#
# using Cordero covalent radii, so r is about 1 at a relaxed bond, below 1 when
# compressed and above 1 when stretched. The smallest r over the shell (rmin) is
# binned by DIST_RATIO_MIN_EDGES into five states: two compression bands, a
# near-equilibrium band, and a stretched band.
#
# Sites with no usable neighbour set, or whose neighbours have no tabulated
# covalent radius, get DIST_BIN_UNDEFINED so they all collapse into one class
# instead of fragmenting. 999 cannot collide with a real bin index.
#
# DIST_RATIO_MAX_EDGES is kept so the stored rmax stays re-binnable, but rmax is
# not part of the token: chemenv picks the neighbour shell with a cutoff relative
# to the nearest contact, so a large rmax means a wide shell rather than a
# stretched bond.
DIST_RATIO_MIN_EDGES = (0.60, 0.70, 0.85, 1.15)
DIST_RATIO_MAX_EDGES = (0.85, 1.15, 1.50)
DIST_BIN_UNDEFINED = 999

# ---------------------------------------------------------------------------
# The asymmetry field A.
#
#     u_ij = d_ij / (R_cov[i] + R_cov[j])      the same reduced length as above
#     w(u) = exp(-(u - 1) / A_LAMBDA)          1 at contact, larger as u falls
#     A_i  = | sum_j w(u_ij) * rhat_ij |       magnitude of the vector sum
#
# A measures how one-sided a site's contacts are. Force is a vector sum of pair
# terms, but every other field in the token is a scalar or an unordered set: a
# symmetric compressed shell and the same contacts pushed to one side look
# identical to them, yet only the second carries a net force. A separates them.
#
# A_LAMBDA sets how sharply the weighting favours short contacts. Small values
# make A collapse onto the single shortest contact, i.e. a finer rmin; large
# values wash the compression weighting out.
#
# A cannot overflow. u is always >= 0 (contacts with d <= 0 are skipped), so
#     w(u) < exp(1 / A_LAMBDA) = e^10
# and A stays below n_neighbours * e^10. np.exp here can never return inf.
A_LAMBDA = 0.10
A_EDGES = (1.0, 3.0)

#Highest Z for which Cordero et al. (2008) actually report a covalent radius.
#ase.data.covalent_radii is indexed all the way to Z = 118, but Z = 97..118 are
#PADDED with a 2.0 A placeholder rather than left as NaN, so a bound of
#len(covalent_radii) alone would accept a fabricated radius. See _covalent_radius().
CORDERO_MAX_Z = 96


def _covalent_radius(symbol):
    """Cordero covalent radius in Angstrom, or None if the element is not tabulated.

    The Z <= CORDERO_MAX_Z bound is what actually makes the "not tabulated" branch
    reachable: ASE pads Z = 97..118 with a 2.0 A placeholder instead of NaN, so
    without it a transactinide would be silently binned against a fabricated
    reference distance rather than reported undefined. This is a no-op for
    OMat24/MatPES -- nothing above Cm (96) occurs in either, and Bk..Lr (97..103)
    are dropped upstream by contains_rare_earths() anyway -- so no existing site's
    bins change. It only keeps DIST_BIN_UNDEFINED honest on future data.
    """
    z = atomic_numbers.get(symbol)
    if z is None or z <= 0 or z > CORDERO_MAX_Z or z >= len(covalent_radii):
        return None
    r = float(covalent_radii[z])
    if not np.isfinite(r) or r <= 0.0:
        return None
    return r


def _distance_ratio_bins(structure, site_index, neighbor_sites):
    """Return (ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin) for one site.

    _and_A: the asymmetry A is accumulated INSIDE the same loop, over the same
    shell, under the same skip rules, so A can never describe a different set of
    contacts from the ones that produced ratio_min/ratio_max. That is also why the
    CrystalNN rescue path gets A for free -- it reuses this helper unchanged.

    `neighbor_sites` MUST be the same neighbour objects that produced
    coord_elements, so that the distance fields and the composition field can
    never describe two different shells. Each entry only needs `.coords`
    (cartesian and image-resolved, i.e. already unwrapped across the periodic
    boundary) and `.specie.symbol` -- true both for chemenv's neighb_sites and
    for the sites returned by CrystalNN.get_nn_info, which is how the rescue path
    below can reuse this helper unchanged.

    Distances are measured the same way as in _neighbor_keyset(): the norm of the
    neighbour image coordinate minus the central site coordinate. Contacts whose
    element has no tabulated radius are skipped; if that leaves no usable contact
    the site is reported undefined (NaN ratios, 999 bins) instead of being
    silently binned from a partial shell.
    """
    r_central = _covalent_radius(structure[site_index].specie.symbol)
    ratios = []
    #Weighted sum of UNIT vectors from the central site to each usable contact.
    #Its magnitude is A; it stays exactly zero for a centrosymmetric shell.
    a_vec = np.zeros(3, dtype=float)
    if r_central is not None:
        center = np.array(structure[site_index].coords)
        for nb in neighbor_sites:
            r_nb = _covalent_radius(nb.specie.symbol)
            if r_nb is None:
                continue
            d0 = r_central + r_nb
            #vec hoisted out of the norm: A needs the direction as well as the length
            vec = np.array(nb.coords) - center
            dist = float(np.linalg.norm(vec))
            if not np.isfinite(dist) or dist <= 0.0:
                #Guard against a degenerate self-contact / duplicated site.
                continue
            ratio = dist / d0
            ratios.append(ratio)
            a_vec = a_vec + np.exp(-(ratio - 1.0) / A_LAMBDA) * (vec / dist)
    if not ratios:
        #A is undefined for exactly the sites whose ratios are, and shares the
        #999 sentinel so they all collapse into ONE class rather than fragmenting.
        return (float(np.nan), float(np.nan), DIST_BIN_UNDEFINED, DIST_BIN_UNDEFINED,
                float(np.nan), DIST_BIN_UNDEFINED)
    ratio_min = min(ratios)
    ratio_max = max(ratios)
    a_raw = float(np.linalg.norm(a_vec))
    return (float(ratio_min), float(ratio_max),
            int(np.digitize(ratio_min, DIST_RATIO_MIN_EDGES)),
            int(np.digitize(ratio_max, DIST_RATIO_MAX_EDGES)),
            a_raw, int(np.digitize(a_raw, A_EDGES)))


def _neighbor_keyset(structure, site_index, neighbors_set):
    """Multiset of (element, distance-rounded-to-0.01-A) for a chemenv neighbour set."""
    center = np.array(structure[site_index].coords)
    keys = Counter()
    for nb in neighbors_set.neighb_sites:
        dist = round(float(np.linalg.norm(np.array(nb.coords) - center)), 2)
        keys[(nb.specie.symbol, dist)] += 1
    return keys


def select_site_lse(lse_default, lse_extended, structure, site_index, vnn,
                    threshold=VORONOI_PROMINENCE_THRESHOLD):
    """Pick the 1.4 or 1.7 LightStructureEnvironments for one site (see block header).

    Returns lse_extended iff the extended neighbour set adds at least one contact
    whose normalised VoronoiNN solid-angle weight is >= `threshold`; otherwise
    returns lse_default. Falls back to lse_default on any failure (missing
    extended set, Voronoi construction error, etc.) so the selector can never make
    a site worse than the default-cutoff behaviour of the original script.
    """
    if lse_default is None or lse_extended is None:
        return lse_default
    try:
        ns_default = lse_default.neighbors_sets[site_index]
        ns_extended = lse_extended.neighbors_sets[site_index]
    except Exception:
        return lse_default
    if not ns_extended:
        return lse_default
    k_default = _neighbor_keyset(structure, site_index, ns_default[0]) if ns_default else Counter()
    k_extended = _neighbor_keyset(structure, site_index, ns_extended[0])
    #Neighbours present in the 1.7 set beyond the 1.4 set.
    added = [k for k in (set(k_extended) | set(k_default))
             if k_extended.get(k, 0) > k_default.get(k, 0)]
    if not added:
        return lse_default
    try:
        center = np.array(structure[site_index].coords)
        weight_map = {}
        for n in vnn.get_nn_info(structure, site_index):
            dist = round(float(np.linalg.norm(np.array(n['site'].coords) - center)), 2)
            key = (n['site'].specie.symbol, dist)
            if n['weight'] > weight_map.get(key, 0.0):
                weight_map[key] = n['weight']
    except Exception:
        return lse_default
    prominence = max((weight_map.get(k, 0.0) for k in added), default=0.0)
    return lse_extended if prominence >= threshold else lse_default


def build_coord_descriptor(lse, structure, site_index, cnn):
    """Return (coord_elements, coord_symbol, sym_measure, perturb_scale,
    ratio_min, ratio_max, dist_bin_min, dist_bin_max, a_raw, a_bin) for a site.
    Note dist_bin_max is still COMPUTED and returned; it is simply no longer part of
    the env_id -- a_bin took its slot.

    This is the single source of truth for the coordination part of the env_id.
    NOTE: unlike the earlier 4-tuple version of this function
    this returns 10 values — the four original fields, the raw min/max distance
    ratios and their bins, and (_and_A) the raw asymmetry and its bin. This function
    is identical in behaviour to the one in
    stage 01's tokenize_omat24.py -- keep the two in sync.

    The distance ratios are computed from coord_info['neighbor_sites'], i.e. the
    very shell whose elements form coord_elements, so composition and distance
    always refer to the same neighbour set. The CrystalNN rescue path below does
    the same with its own neighbour list.

    Primary source is the chemenv coordination environment (coord_symbol like
    'O:6', distortion-bucketed perturb_scale from the continuous symmetry measure).
    When chemenv assigns nothing for the site — no environment, or the whole
    structure's Voronoi construction failed (common for 2D / vacuum-containing
    cells, where chemenv raises and lse is None) — we fall back to CrystalNN so the
    neighbour composition and coordination number are still recorded instead of
    being thrown away. Such rescued sites are labelled with chemenv's OWN
    'UNKNOWN:<cn>' symbol (the same token chemenv emits when it finds a neighbour
    set but matches no ideal geometry) with perturb_scale = 999, so that every
    'CN n, no geometry' atom falls into ONE class for classification regardless of
    whether chemenv or CrystalNN supplied the neighbours. Only if CrystalNN also
    fails is the descriptor fully INVALID.
    """
    coord_info = get_complete_coordination_info(lse, structure, site_index)
    coord_env = coord_info['coordination_environments']
    if coord_env != 'INVALID' and coord_env is not None and len(coord_env) > 0:
        coord_symbol = coord_env[0]['ce_symbol']
        sym_measure = coord_env[0]['csm']
        coord_elements = "-".join(sorted(list(coord_info['neighbor_elements'])))
        if sym_measure is not None:
            if sym_measure <= 1.0:
                perturb_scale = 0
            elif sym_measure <= 5:
                perturb_scale = 1
            elif sym_measure < 10:
                perturb_scale = 2
            else:
                perturb_scale = 3
        else:
            perturb_scale = 999
            sym_measure = float(np.nan)
        ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin = _distance_ratio_bins(
            structure, site_index, coord_info['neighbor_sites'])
        return (coord_elements, coord_symbol, float(sym_measure), perturb_scale,
                ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin)

    #Rescue: chemenv could not assign an environment. Recover the neighbour
    #composition + coordination number with CrystalNN (robust to the low-dimensional
    #/ pathological-Voronoi cases that defeat chemenv).
    try:
        nn_info = cnn.get_nn_info(structure, site_index)
        neigh = sorted(n['site'].specie.symbol for n in nn_info)
        if len(neigh) > 0:
            ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin = _distance_ratio_bins(
                structure, site_index, [n['site'] for n in nn_info])
            return ("-".join(neigh), f"UNKNOWN:{len(neigh)}", float(np.nan), 999,
                    ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin)
    except Exception:
        pass

    return ('INVALID', 'INVALID', float(np.nan), 999,
            float(np.nan), float(np.nan), DIST_BIN_UNDEFINED, DIST_BIN_UNDEFINED,
            float(np.nan), DIST_BIN_UNDEFINED)

def process_single_structure_chemenv(s):
    """Return a list of chemenv ids, one per atom in `s` (original ASE ordering).

    Fixes relative to v2:
      * Spin is taken from each atom's OWN initial magnetic moment. v2 read the
        spin off a `find_primitive(keep_site_properties=True)` structure, but
        that call broadcasts the first site-of-each-species magmom to every site
        of that species, so all e.g. Mn atoms inherited a single spin value.
      * Coordination environments are computed directly on the ORIGINAL cell,
        restricted (via `only_indices`) to one representative per symmetry orbit.
        v2 mixed two inconsistent primitive cells: `mapping_to_primitive` indexes
        spglib's standardized primitive, while the environments were read from
        `sga.find_primitive()` — a different construction with no guaranteed
        matching site order, which could mis-assign environments among
        same-element sites for non-P1 structures.
      * The LocalGeometryFinder is set up with structure_refinement="none" (see
        main()). With the default ("refined") chemenv silently computes on
        get_refined_structure() of the input, so the lse indices it returns do
        NOT line up with the structure we pass — corrupting coordination labels
        for centered (I/F/C) Bravais lattices. "none" keeps indices aligned 1:1.
        Note: as a side effect the continuous symmetry measure (csm -> perturb
        scale) now reflects the actual, non-idealised geometry of each structure.
    """

    #Read in structure into ASE object
    atoms_formula = s.get_chemical_formula()
    if contains_rare_earths(s):
        #Skip if structure contains rare earth elements
        print(f'Skipping {atoms_formula}. Rare-earth elements.', flush=True)
        return None

    #Get atomic magnetic moments and material ID from OMat24
    magmoms = s.get_initial_magnetic_moments()
    mat_id = s.info['sid']

    #Convert ASE to pymatgen structure (ordering is preserved 1:1 with ASE)
    pymatgen_atoms = Structure.from_ase_atoms(s)
    pymatgen_atoms.add_site_property("magmom", magmoms)

    #Symmetry-equivalent atoms are defined on the ORIGINAL structure's indices.
    #Atoms in the same orbit share an identical coordination environment, so we
    #only need to analyse one representative per orbit.
    #Grouping by symmetry is purely an optimisation (analyse one site per orbit,
    #broadcast to equivalents). spglib RAISES (e.g. SymmetryUndeterminedError on
    #near-degenerate cells) or may return None — in either case fall back to
    #treating every atom as its own representative, i.e. analyse every site with
    #no broadcasting. This must never abort the run.
    n_sites = len(pymatgen_atoms)
    try:
        sym_dataset = SpacegroupAnalyzer(pymatgen_atoms).get_symmetry_dataset()
        if sym_dataset is None:
            raise ValueError("spglib returned no symmetry dataset")
        equivalent_atoms = [int(x) for x in sym_dataset.equivalent_atoms]
        if len(equivalent_atoms) != n_sites:
            raise ValueError(f"equivalent_atoms length {len(equivalent_atoms)} != {n_sites} sites")
    except Exception as e:
        print(f'Warning: symmetry analysis failed for {mat_id} ({atoms_formula}); '
              f'analysing every site individually: {e}', flush=True)
        equivalent_atoms = list(range(n_sites))
    unique_reps = list(dict.fromkeys(equivalent_atoms))

    #Feed the ORIGINAL structure to LocalGeometryFinder(), computing environments
    #only for the representative sites (neighbour search still uses the full cell).
    #Build the structure environments ONCE (the expensive step), then SELECT among
    #them with two cheap strategies: the default 1.4 cutoff and the extended 1.7
    #cutoff (keyword args — the first positional arg of SimplestChemenvStrategy is
    #structure_environments, not distance_cutoff). The per-atom choice between them
    #is made later by select_site_lse(). maximum_distance_factor defaults to 2.0,
    #which comfortably reaches the 1.7 relative cutoff.
    lgf.setup_structure(pymatgen_atoms)
    try:
        se = lgf.compute_structure_environments(only_cations=False, only_indices=unique_reps)
        lse_default = LightStructureEnvironments.from_structure_environments(
            strategy=SimplestChemenvStrategy(distance_cutoff=DISTANCE_CUTOFF_DEFAULT, angle_cutoff=ANGLE_CUTOFF),
            structure_environments=se)
        lse_extended = LightStructureEnvironments.from_structure_environments(
            strategy=SimplestChemenvStrategy(distance_cutoff=DISTANCE_CUTOFF_EXTENDED, angle_cutoff=ANGLE_CUTOFF),
            structure_environments=se)
    except Exception as e:
        print(f'Warning: chemenv failed for {mat_id} ({atoms_formula}): {e}', flush=True)
        lse_default = None
        lse_extended = None

    print(f'Processing material {mat_id} ({atoms_formula})', flush=True)

    #Build the coordination descriptor (neighbours + geometry + distortion) for
    #each representative site via the shared helper (chemenv, with CrystalNN rescue
    #for sites/structures chemenv cannot handle). Spin is intentionally NOT included
    #here — it is a per-atom property handled in the expansion loop below.
    cnn = CrystalNN()
    vnn = VoronoiNN(tol=0.0)
    rep_coord_desc = {}
    for rep in unique_reps:
        chosen_lse = select_site_lse(lse_default, lse_extended, pymatgen_atoms, rep, vnn)
        (coord_elements, coord_symbol, sym_measure, perturb_scale,
         dist_ratio_min, dist_ratio_max, dist_bin_min, dist_bin_max,
         a_raw, a_bin) = build_coord_descriptor(
            chosen_lse, pymatgen_atoms, rep, cnn)
        #Only the fields that go into env_id are kept, as before -- sym_measure,
        #the raw ratios and raw A are discarded here (the database-side script
        #stores them as HDF5 columns; this script only ever emits the token).
        #_and_A: dist_bin_max is no longer part of the token, so it is dropped
        #here too; a_bin takes its place as the final field.
        rep_coord_desc[rep] = (coord_elements, coord_symbol, perturb_scale,
                               dist_bin_min, a_bin)

    #Expand to every atom: element + its OWN rounded |magmom| + representative's
    #coordination descriptor.
    env_ids_per_atom = []
    for i in range(len(pymatgen_atoms)):
        rep = equivalent_atoms[i]
        central_element = pymatgen_atoms[i].specie.symbol
        spin = round(np.abs(magmoms[i]))
        (coord_elements, coord_symbol, perturb_scale,
         dist_bin_min, a_bin) = rep_coord_desc[rep]
        #Field order MUST match stage 01's tokenize_omat24.py:
        #  <element>_<spin>_<coord_elements>_<coord_symbol>_<perturb_scale>_<rmin_bin>_<a_bin>
        #The env_id IS the join key between the two sides, so any divergence here is
        #silent: it does not raise, it just fails to match. Tested by
        #test_and_A_downstream.py, which builds both tokens from the same structures
        #and requires them to be byte-identical.
        env_id = (f"{central_element}_{spin}_{coord_elements}_{coord_symbol}_{perturb_scale}"
                  f"_{dist_bin_min}_{a_bin}")
        env_ids_per_atom.append(env_id)

    return env_ids_per_atom


# ----------------------------------------------------------------------------
# Random-access extxyz reading via a byte-offset index.
#
# build_frame_offsets() does ONE sequential pass over the file, recording the
# byte offset of every frame's natoms line (it only counts lines, never parses
# coordinates, so it is ~1 us/frame). read_frame_at() then seeks straight to a
# given offset and parses exactly that one frame. Together they replace
# ase.io.read(file, index=idx), which re-scans the whole file on every call.
# Each MPI rank builds the offsets itself: the pass is cheap, gives n_structures
# for free, and avoids broadcasting a multi-MB index. An extxyz frame is
# <natoms>\n <comment>\n <natoms atom lines>, with no blank separators.
# ----------------------------------------------------------------------------
def build_frame_offsets(path):
    """Return a list of byte offsets, one per frame's natoms line."""
    offsets = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue  # tolerate stray blank lines between frames
            try:
                nat = int(s)
            except ValueError:
                raise RuntimeError(
                    f"{path}: expected an integer natoms line at byte {pos}, "
                    f"got {s[:60]!r}")
            offsets.append(pos)
            f.readline()                      # comment / Lattice line
            for _ in range(nat):
                f.readline()                  # atom lines
    return offsets


def read_frame_at(path, offset):
    """Seek to `offset` and parse exactly one extxyz frame into an ASE Atoms."""
    with open(path, "rb") as f:
        f.seek(offset)
        nat = int(f.readline())
        rest = [f.readline() for _ in range(nat + 1)]   # comment + atom lines
        text = str(nat) + "\n" + b"".join(rest).decode()
    # parallel=False is ESSENTIAL: ase.io.read defaults to parallel=True, which
    # under a >1-rank MPI world makes ONLY rank 0 parse and BROADCASTS its result
    # to every rank — so every rank would silently receive rank 0's structure
    # instead of the frame it seeked to. parallel=False forces each rank to parse
    # its own text locally.
    return ase.io.read(StringIO(text), format="extxyz", index=0, parallel=False)


def process_chunk(indices, offsets):
    """
    Each worker calls this: seeks directly to its assigned structures via the
    byte-offset index, processes them, and returns lightweight results.
    """

    results = []
    for idx in indices:
        # Random-access read of a single frame (O(1) — no full-file scan).
        atoms = read_frame_at(MACE_PRED_EXTXYZ, offsets[idx])

        if BANNED_ELEMENTS & set(atoms.get_chemical_symbols()):
            print(f"Excluding {atoms.get_chemical_formula()}", flush=True)
            continue

        n_atoms = len(atoms)
        f_ref = atoms.arrays['REF_forces'] * 1000.0
        f_mace = atoms.arrays['MACE_forces'] * 1000.0
        f_diff = f_mace - f_ref                           # (N_atoms, 3) component-wise
        f_ae_mace = np.linalg.norm(f_diff, axis=1)         # (N_atoms,)   per-atom norm

        env_ids_per_atom = process_single_structure_chemenv(atoms)
        if env_ids_per_atom is None:
            continue

        atoms.arrays['chemenv_id'] = np.array(env_ids_per_atom, dtype=object)
        atoms.arrays['f_ae_mace'] = f_ae_mace
        results.append((atoms, f_ae_mace, f_diff))

    return results  # list of (atoms, f_ae_mace, f_diff) for this chunk




def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Step 1: every rank builds the byte-offset index itself (cheap sequential
    #pass, ~1 us/frame) and so learns n_structures locally — no count broadcast
    #needed. The offsets give O(1) random access to any frame below.
    t0 = time.time()
    offsets = build_frame_offsets(MACE_PRED_EXTXYZ)
    n_structures = len(offsets)
    if rank == 0:
        print(f'Rank 0: indexed {n_structures} structures in '
              f'{(time.time()-t0)/60:.2f} min '
              f'({len(offsets)*8/1e6:.1f} MB of offsets)', flush=True)

    #Step 2: distribute indices across ranks (round-robin striping; identical
    #assignment to the original script, so per-rank output is unchanged).
    all_indices = list(range(n_structures))
    rank_indices = all_indices[rank::size]

    #Step 3: each rank initialises its own LocalGeometryFinder
    global lgf
    lgf = LocalGeometryFinder()
    #structure_refinement="none" is ESSENTIAL: with the default ("refined"),
    #LocalGeometryFinder.setup_structure() internally replaces our structure with
    #SpacegroupAnalyzer(...).get_refined_structure() (the conventional standard
    #cell). All site indices in the returned LightStructureEnvironments then refer
    #to that internal cell, NOT to the structure we passed. For centered Bravais
    #lattices the conventional cell has more atoms (and a different order) than our
    #input, so `only_indices`/site_index would select the wrong sites. With "none"
    #chemenv uses exactly the structure we hand it, keeping indices aligned 1:1
    #with our atoms and their magmoms.
    lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True,
                         structure_refinement=LocalGeometryFinder.STRUCTURE_REFINEMENT_NONE)
    comm.Barrier()
    print(f'Rank {rank}: starting {len(rank_indices)} structures', flush=True)

    #Step 4: process in chunks. Each chunk is written to this rank's output file
    #immediately and then DROPPED, so peak memory is bounded by CHUNK_SIZE
    #scalar sums rather than by hoarding every per-atom error array.
    rank_output = f"output_rank{rank}.xyz"
    sum_abs_norm = 0.0   # sum of ||F_pred - F_ref|| over atoms      (norm-based)
    sum_sq_norm  = 0.0   # sum of ||.||^2 over atoms
    n_norm       = 0     # number of atoms
    sum_abs_comp = 0.0   # sum of |F_pred - F_ref| over components   (component-wise)
    sum_sq_comp  = 0.0   # sum of (.)^2 over components
    n_comp       = 0     # number of force components (3 * atoms)
    wrote_any = False

    for chunk_start in range(0, len(rank_indices), CHUNK_SIZE):
        chunk = rank_indices[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_results = process_chunk(chunk, offsets)

        chunk_atoms = []
        for atoms_obj, f_ae, f_diff in chunk_results:
            chunk_atoms.append(atoms_obj)
            sum_abs_norm += float(f_ae.sum())
            sum_sq_norm  += float(np.square(f_ae).sum())
            n_norm       += int(f_ae.size)
            sum_abs_comp += float(np.abs(f_diff).sum())
            sum_sq_comp  += float(np.square(f_diff).sum())
            n_comp       += int(f_diff.size)

        if chunk_atoms:
            # append=wrote_any: first write overwrites any stale file, the rest append.
            ase.io.write(rank_output, chunk_atoms, format='extxyz',
                         append=wrote_any, write_info=True, write_results=True,
                         parallel=False)
            wrote_any = True
        # chunk_atoms / chunk_results go out of scope here -> memory released.
        print(f'Rank {rank}: done chunk ending at local idx {chunk_start + len(chunk)}', flush=True)

    #Ensure this rank's output file exists even if it processed nothing, so the
    #concatenation step below is uniform.
    if not wrote_any:
        open(rank_output, 'w').close()

    print(f'Rank {rank} finished processing {n_norm} atoms altogether', flush=True)

    #Step 5: reduce the six scalar accumulators onto rank 0 (O(1) memory) and
    #report global MAE/RMSE. Matches the original definitions to fp epsilon.
    g_sum_abs_norm = comm.reduce(sum_abs_norm, op=MPI.SUM, root=0)
    g_sum_sq_norm  = comm.reduce(sum_sq_norm,  op=MPI.SUM, root=0)
    g_n_norm       = comm.reduce(n_norm,       op=MPI.SUM, root=0)
    g_sum_abs_comp = comm.reduce(sum_abs_comp, op=MPI.SUM, root=0)
    g_sum_sq_comp  = comm.reduce(sum_sq_comp,  op=MPI.SUM, root=0)
    g_n_comp       = comm.reduce(n_comp,       op=MPI.SUM, root=0)

    if rank == 0:
        # Norm-based (Euclidean norm of 3D error vector per atom)
        mae_f_norm  = g_sum_abs_norm / g_n_norm
        rmse_f_norm = (g_sum_sq_norm / g_n_norm) ** 0.5

        # Component-wise (same definition as MACE training log)
        mae_f_comp  = g_sum_abs_comp / g_n_comp
        rmse_f_comp = (g_sum_sq_comp / g_n_comp) ** 0.5

        print(f'=== Force errors (norm-based: ||F_pred - F_ref||) ===')
        print(f'MAE  on atomic forces: {mae_f_norm:.4f} meV/A')
        print(f'RMSE on atomic forces: {rmse_f_norm:.4f} meV/A')
        print(f'=== Force errors (component-wise, MACE convention) ===')
        print(f'MAE  on atomic forces: {mae_f_comp:.4f} meV/A')
        print(f'RMSE on atomic forces: {rmse_f_comp:.4f} meV/A')

    comm.Barrier()

    if rank == 0:
        with open(OUTPUT_XYZ, 'w') as outf:
            for r in range(size):
                fname = f"output_rank{r}.xyz"
                if os.path.exists(fname):
                    with open(fname, 'r') as inf:
                        shutil.copyfileobj(inf, outf)
                    os.remove(fname)
        print(f'Written merged output to {OUTPUT_XYZ}')
    print(f'Rank {rank}: finished.', flush=True)

if __name__ == "__main__":
    main()
