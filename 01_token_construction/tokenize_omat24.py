
# REPO CHANGE: fairchem is imported lazily, inside the ase-db branch of main().
# It is only needed to read the OMat24 ASE databases, and it drags in torch +
# torch-geometric -- so requiring it up front would force an xyz-only user to
# install the whole fairchem stack for code they never reach.
from pymatgen.core import Structure
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import Counter

import ase.io
from ase.data import covalent_radii, atomic_numbers
from io import StringIO
import pandas as pd
import os
from mpi4py import MPI
import h5py
import numpy as np
from mpi4py.futures import MPICommExecutor

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK (or set the environment variables) <<<
#
# Every value here is read from the environment so that submit_tokenize.sh can
# drive an 11-element job array without editing the source. Set them in your
# shell (or in the submit script) rather than hardcoding paths below.
#
#   OMAT_DATA_DIR    REQUIRED. The directory that HOLDS THE 11 OMAT24 TRAINING
#                    SUB-DATASETS listed in subsets.txt -- not one sub-dataset,
#                    and not a single file. Each sub-dataset is looked up inside
#                    it as either
#                        $OMAT_DATA_DIR/<subset>/       -> ASE database dir
#                        $OMAT_DATA_DIR/<subset>.xyz    -> extxyz file
#                    e.g. OMAT_DATA_DIR=/data/omat24/train_extracted, holding
#                    rattled-300/, rattled-relax/, ... or rattled-300.xyz, ...
#                    Every frame must carry a periodic cell and info['sid'];
#                    per-atom initial_magmoms are used where present (spin 0
#                    otherwise) -- run check_input.py first.
#                    There is deliberately NO default: a wrong data directory
#                    would waste hundreds of node-days.
#
#   OMAT_INPUT_FORMAT   'auto' (default), 'ase-db' or 'extxyz'. 'auto' decides
#                    per sub-dataset and refuses if a sub-dataset exists in both
#                    forms. Note: compressed .xyz.gz / .extxyz.gz are NOT
#                    supported by the tokeniser -- decompress first.
#
#   OMAT_FILE_TEMPLATE  Only if your extxyz files are not named <subset>.xyz.
#                    A pattern with a {subset} placeholder, applied inside
#                    $OMAT_DATA_DIR, e.g. 'all_structures_{subset}.xyz'.
#
#   OMAT_DB_INDEX    Which line of subsets.txt to process, 0-based (default 0).
#                    submit_tokenize.sh sets it from $SLURM_ARRAY_TASK_ID.
#
# OUTPUT goes into the CURRENT WORKING DIRECTORY: chemenv_<subset>_rank<N>.hdf5,
# one file per MPI rank. Run each sub-dataset in its own directory named exactly
# as in subsets.txt -- stage 01b derives the sub-dataset label from that name.
# ============================================================================
#Requested backend; resolved to a concrete 'ase-db'/'extxyz' in main().
INPUT_FORMAT = os.environ.get('OMAT_INPUT_FORMAT', 'auto').lower()
#Which line of subsets.txt this run processes.
DB_INDEX = int(os.environ.get('OMAT_DB_INDEX', '0'))


def data_dir():
    """The directory holding the 11 sub-datasets ($OMAT_DATA_DIR). Raises if unset.

    REPO CHANGE: this used to fall back to the author's absolute path on
    MeluXina, which silently pointed every other user at a directory they do not
    have. It is now required.
    """
    value = os.environ.get('OMAT_DATA_DIR')
    if not value:
        raise RuntimeError(
            "OMAT_DATA_DIR is not set. Point it at the directory that HOLDS the "
            "sub-datasets named in subsets.txt -- e.g. "
            "export OMAT_DATA_DIR=/data/omat24/train_extracted , which contains "
            "either <subset>/ directories (ASE DB) or <subset>.xyz files. "
            "Run check_input.py --data-dir <that directory> first.")
    return value


#These objects needs to be passed to process_single_structure() but are not pickleable
#Thus defined as global variables
dataset = None

# ============================================================================
# REPO CHANGE -- input backend.
#
# As run, this script read OMat24 through fairchem's AseDBDataset, i.e. from the
# extracted ASE databases. It now reads either those OR extxyz files, whichever
# is actually present for the requested sub-dataset. Everything downstream is
# unchanged, because the only things taken from a frame are the ASE Atoms
# object, its per-atom initial magnetic moments and its info['sid'].
#
# OMAT_INPUT_FORMAT selects the backend:
#   auto (default) -- look for both and use whichever exists; error if the
#                     sub-dataset is available in neither or in BOTH (an
#                     ambiguity worth resolving deliberately, since the two can
#                     hold different data)
#   ase-db         -- OMAT_DATA_DIR/<subset>/      (fairchem AseDBDataset)
#   extxyz         -- OMAT_DATA_DIR/<subset>.xyz   (or .extxyz, or whatever
#                     OMAT_FILE_TEMPLATE says)
#
# The offset-index reader below is the same one used by the prediction-side
# tokeniser in stage 10.2 -- including parallel=False, which is ESSENTIAL: ASE
# defaults to parallel=True, which under a >1-rank MPI world makes only rank 0
# parse and broadcasts its result, so every rank would silently receive rank
# 0's structure instead of the frame it seeked to.
# ============================================================================
#Resolved to a concrete 'ase-db'/'extxyz' in main() on EVERY rank, before the
#executor starts, so workers see the resolved value too. The requested value is
#read in the USER CONFIGURATION block at the top of this file.
frame_offsets = None
#The sub-dataset name from subsets.txt; set by main() on every rank.
SUBSET_NAME = None

#Backend discovery lives in input_backend.py so that check_input.py -- which
#predicts what this script will do, using only the stdlib + ASE -- cannot drift
#from it. sys.path[0] is this file's directory, so the import works wherever the
#job is launched from.
from input_backend import EXTXYZ_SUFFIXES, resolve_input


def build_frame_offsets(path):
    """Return a list of byte offsets, one per extxyz frame's natoms line.

    One sequential pass; lines are counted, never parsed (~1 us/frame).
    """
    offsets = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            t = line.strip()
            if not t:
                continue                      # tolerate stray blank lines
            try:
                nat = int(t)
            except ValueError:
                raise RuntimeError(
                    f"{path}: expected an integer natoms line at byte {pos}, "
                    f"got {t[:60]!r}")
            offsets.append(pos)
            f.readline()                      # comment / Lattice line
            for _ in range(nat):
                f.readline()                  # atom lines
    return offsets


def read_frame_at(path, offset):
    """Seek to `offset` and parse exactly one extxyz frame into an ASE Atoms.

    parallel=False is ESSENTIAL: ase.io.read defaults to parallel=True, which
    under a >1-rank MPI world makes ONLY rank 0 parse and BROADCASTS its result,
    so every rank would silently receive rank 0's structure instead of the frame
    it seeked to.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        nat = int(f.readline())
        rest = [f.readline() for _ in range(nat + 1)]   # comment + atom lines
        text = str(nat) + "\n" + b"".join(rest).decode()
    return ase.io.read(StringIO(text), format="extxyz", index=0, parallel=False)


def get_structure(s, dataset_path):
    """Return frame `s` of `dataset_path` as an ASE Atoms, whatever the backend."""
    if INPUT_FORMAT == 'extxyz':
        return read_frame_at(dataset_path, frame_offsets[s])
    return dataset.get_atoms(s)


def dataset_label(dataset_path):
    """Sub-dataset name used in output filenames and the omat_subset column.

    This is the name from subsets.txt, set by main() on every rank, so the two
    backends label a sub-dataset identically and OMAT_FILE_TEMPLATE cannot leak
    a decorated filename into the label (stage 01b derives the same label from
    the working-directory name, and the two must agree).

    Falls back to deriving it from the path when the global is unset, e.g. if
    something imports this module without going through main().
    """
    if SUBSET_NAME is not None:
        return SUBSET_NAME
    base = os.path.basename(os.path.normpath(dataset_path))
    for ext in EXTXYZ_SUFFIXES:
        if base.endswith(ext):
            return base[:-len(ext)]
    return base
lgf = None

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
# These four constants also appear in stage 10.2's analyze_predictions.py, which
# has to produce the same tokens. Keep the two copies in sync.
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
    Note dist_bin_max is still COMPUTED and returned (and stored as an HDF5 column);
    it is simply no longer part of the env_id -- a_bin took its slot.

    This is the single source of truth for the coordination part of the env_id.
    It returns 10 values: the four coordination fields, the raw min/max distance
    ratios and their bins, and the raw asymmetry and its bin. Stage 10.2 has the
    same function and must return the same values, or tokens from the two sides
    stop being comparable.

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

def contains_rare_earths(structure):
    rare_earth_elements = {'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'} 
    if bool(set(structure.get_chemical_symbols()).intersection(rare_earth_elements)):
        return True
    else:
        return False


#def process_single_structure(dataset, lgf, dataset_path, min_itemsize, s):
def process_single_structure(s, dataset_path, min_itemsize):
    rank = MPI.COMM_WORLD.Get_rank()
    local_hdf5_filename=f"chemenv_{dataset_label(dataset_path)}_rank{rank}.hdf5"

    #This list stores the chemical environments of all sites in the structure
    env_list = []

    #Read in structure into ASE object
    atoms = get_structure(s, dataset_path)
    atoms_formula = atoms.get_chemical_formula()
    if contains_rare_earths(atoms):
        #Skip if structure contains rare earth elements
        print(f'Skipping {atoms_formula}. Rare-earth elements.', flush=True)
        return None

    #Get atomic magnetic moments and material ID from OMat24 
    magmoms = atoms.get_initial_magnetic_moments()
    mat_id = atoms.info['sid']

    #Convert ASE to pymatgen object (ordering preserved 1:1 with ASE / magmoms)
    pymatgen_atoms = Structure.from_ase_atoms(atoms)
    pymatgen_atoms.add_site_property("magmom", magmoms)

    #Symmetry-equivalent atoms are defined on the ORIGINAL structure's indices.
    #Atoms in the same orbit share an identical coordination environment, so we
    #only analyse one representative per orbit (but every atom keeps its OWN spin).
    #
    #Three things this relies on:
    #  * Spin comes from each atom's own magmom. Taking it from a primitive cell
    #    would broadcast one value to every site of a species, so e.g. all Mn
    #    would share a spin.
    #  * Environments are computed on the ORIGINAL cell, with
    #    structure_refinement="none". Any refinement replaces the cell internally,
    #    and site indices then no longer match the structure that was passed in.
    #  * One row is emitted per ATOM, so per-atom spins survive and environment
    #    frequencies carry the right multiplicities.
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
        print(f'Rank {rank}: Warning: symmetry analysis failed for {mat_id} ({atoms_formula}); '
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
        print(f'Rank {rank}: Warning: chemenv failed for {mat_id} ({atoms_formula}): {e}', flush=True)
        lse_default = None
        lse_extended = None

    print(f'Rank {rank} processing material {mat_id} ({atoms_formula})', flush=True)

    #Coordination descriptor (neighbours + geometry + distortion) per representative
    #via the shared helper (chemenv, with CrystalNN rescue for sites/structures
    #chemenv cannot handle). Spin is NOT included here; it is a per-atom property
    #added in the expansion below.
    cnn = CrystalNN()
    vnn = VoronoiNN(tol=0.0)
    rep_coord_desc = {}
    for rep in unique_reps:
        chosen_lse = select_site_lse(lse_default, lse_extended, pymatgen_atoms, rep, vnn)
        rep_coord_desc[rep] = build_coord_descriptor(chosen_lse, pymatgen_atoms, rep, cnn)

    #Expand to every atom: element + its OWN rounded |magmom| + representative's
    #coordination descriptor. One row per atom.
    for site_index in range(len(pymatgen_atoms)):
        central_element = pymatgen_atoms[site_index].specie.symbol
        spin = round(np.abs(magmoms[site_index]))
        (coord_elements, coord_symbol, sym_measure, perturb_scale,
         dist_ratio_min, dist_ratio_max, dist_bin_min, dist_bin_max,
         a_raw, a_bin) = rep_coord_desc[equivalent_atoms[site_index]]

        #_and_A: the LAST field is the asymmetry bin, NOT dist_bin_max. The token
        #has the same NUMBER of fields as before, so a v12/v13 parser will read it
        #without error and silently take A for dmax -- exactly the mis-parse the
        #header of the subset script warns about. Downstream scripts MUST be
        #updated; there is deliberately no in-band marker to catch it for you.
        #
        #That field count is NOT a constant: 11 chemenv polyhedron symbols contain
        #an underscore themselves ('BO_1:8', 'TT_2:9', 'BS_1:10', ...), so a token
        #may split into 7 OR 8 parts. Any parser must index the trailing fields
        #from the END (a_bin = parts[-1], dmin = parts[-2], perturb = parts[-3]) or
        #anchor on the ':' of the polyhedron symbol. Splitting and taking parts[4]
        #is wrong for those 11 symbols -- and wrong SILENTLY.
        env_id = (central_element + '_' + str(spin) + '_' + coord_elements + '_' + coord_symbol
                  + '_' + str(perturb_scale) + '_' + str(dist_bin_min) + '_' + str(a_bin))

        env_list.append(
                {'env_id': env_id,
                 'central_element' : str(central_element),
                 'spin' : int(spin),
                 'coord_symbol' : str(coord_symbol),
                 'coord_elements' : str(coord_elements),
                 'sym_measure' : float(sym_measure),
                 #Raw ratios kept alongside the binned values in env_id, mirroring how
                 #sym_measure is kept alongside the bucketed perturb_scale: this allows
                 #re-binning later without re-running the (expensive) chemenv pass.
                 'dist_ratio_min' : float(dist_ratio_min),
                 'dist_ratio_max' : float(dist_ratio_max),
                 #_and_A: raw A kept alongside the binned value for the same reason
                 #the raw ratios are -- re-binning later without re-running chemenv.
                 #dist_ratio_max/dist_bin_max are still stored even though dmax has
                 #left the token, so dropping it stays reversible.
                 'a_raw' : float(a_raw),
                 'a_bin' : int(a_bin),
                 'dist_bin_max' : int(dist_bin_max),
                 'isite' : int(site_index),
                 'mat_id' : str(mat_id),
                 'omat_subset' : str(dataset_label(dataset_path)),
                 'omat_subset_idx' : int(s)
                 }
                )



    batch_df = pd.DataFrame(env_list)
    if os.path.exists(local_hdf5_filename):
        # Append to existing file (without header)
        batch_df.to_hdf(local_hdf5_filename, mode='a', append=True, format='table', key='data', index=False, min_itemsize=min_itemsize)#, header=False, index=False)
        print(f"Batch: Appended {len(batch_df)} results to {local_hdf5_filename}", flush=True)
    else:
        batch_df.to_hdf(local_hdf5_filename, mode='w', format='table', key='data', index=False, min_itemsize=min_itemsize)#, header=True, index=False)
        print(f"Batch: Created {local_hdf5_filename} with {len(batch_df)} results", flush=True)

    #Save the last structure index s written to the file in the metadata
    #with pd.HDFStore(local_hdf5_filename, mode='a') as store:

    return None













#The OMat24 training sub-datasets, in canonical order. Read from subsets.txt so
#that this script and submit_tokenize.sh cannot drift apart; the list below is
#the fallback if that file is missing.
_DEFAULT_SUBSETS = [
    "aimd-from-PBE-1000-npt",
    "aimd-from-PBE-1000-nvt",
    "aimd-from-PBE-3000-npt",
    "aimd-from-PBE-3000-nvt",
    "rattled-1000",
    "rattled-1000-subsampled",
    "rattled-300",
    "rattled-300-subsampled",
    "rattled-500",
    "rattled-500-subsampled",
    "rattled-relax",
]


def _subsets():
    from input_backend import load_subsets
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subsets.txt")
    return load_subsets(path) if os.path.isfile(path) else list(_DEFAULT_SUBSETS)


SUBSETS = _subsets()


def main():
    # All user-settable values come from the USER CONFIGURATION block at the top
    # of this file; nothing site-specific is hardcoded here.
    global INPUT_FORMAT, SUBSET_NAME
    db_dir = data_dir()

    config_kwargs = {} # see tutorial on additional configuration

    # REPO CHANGE (was `db_index = 0`): the 11 OMat24 sub-datasets were originally
    # processed by editing this constant and resubmitting 11 times. Reading it from
    # the environment makes the same 11 runs a single SLURM job array. Unset -> 0,
    # i.e. byte-identical behaviour to the as-run script.
    db_index = DB_INDEX
    if not 0 <= db_index < len(SUBSETS):
        raise IndexError(
            f"OMAT_DB_INDEX={db_index} is out of range for the {len(SUBSETS)} "
            f"sub-datasets listed in subsets.txt.")
    subset_name = SUBSET_NAME = SUBSETS[db_index]

    #Resolve the backend BEFORE the executor starts, so every rank -- including
    #the workers that only ever call process_single_structure -- sees the same
    #concrete value in the INPUT_FORMAT global.
    dataset_path, INPUT_FORMAT = resolve_input(db_dir, subset_name, INPUT_FORMAT)
    #chunk_size = 50 #Number of materials per MPI task to be stored before writing to df
    #DataFrame to save all environments

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Rank {rank} has been initialized")

    global dataset, frame_offsets
    if INPUT_FORMAT == 'extxyz':
        #Rank 0 builds the byte-offset index and broadcasts it. Stage 10.2 has
        #every rank build its own, which is fine for its 657k-frame file; at
        #OMat24 scale that would be `size` full scans of a multi-GB xyz (2560 of
        #them on the production job) for no gain. As an int64 array the index is
        #~8 bytes per frame, so a few tens of MB -- cheap to broadcast and far
        #smaller than the equivalent Python list of ints.
        if rank == 0:
            frame_offsets = np.asarray(build_frame_offsets(dataset_path),
                                       dtype=np.int64)
            print(f"Indexed {len(frame_offsets)} frames in {dataset_path}",
                  flush=True)
        frame_offsets = comm.bcast(frame_offsets if rank == 0 else None, root=0)
        n_samples = len(frame_offsets)
    else:
        try:
            from fairchem.core.datasets import AseDBDataset
        except ImportError as exc:
            raise ImportError(
                f"reading '{dataset_path}' as an ASE database needs fairchem-core "
                f"({exc}). Install it, or point OMAT_DATA_DIR at extxyz files -- "
                f"the extxyz backend needs no fairchem.") from exc
        dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))
        n_samples = dataset.num_samples

    if n_samples == 0:
        raise RuntimeError(f"{dataset_path} contains no structures.")

    print(f"Sub-dataset '{subset_name}' [{INPUT_FORMAT}] -> {dataset_path}", flush=True)
    print(f"Number of entries in {dataset_path}: {n_samples}", flush=True)

    # REPO CHANGE -- input validation. info['sid'] becomes mat_id; without it
    # every frame raises KeyError, so it is checked here on frame 0 instead of
    # 2560 ranks in.
    #Only rank 0 probes, but the VERDICT is broadcast and every rank raises on
    #it. A bare `if rank == 0: raise` in front of a collective would leave the
    #other ranks blocked in the barrier until the walltime ran out.
    check_error = None
    if rank == 0:
        try:
            probe = get_structure(0, dataset_path)
            if 'sid' not in probe.info:
                raise RuntimeError(
                    f"{dataset_path}: frame 0 has no 'sid' in its info dict. "
                    f"mat_id comes from info['sid'], and stage 05 matches the "
                    f"same sid= token in the extxyz comment line. Available "
                    f"keys: {sorted(probe.info)}")
            print(f"Input check passed for {dataset_path} (sid present)", flush=True)
        except Exception as exc:
            check_error = f"{type(exc).__name__}: {exc}"
    check_error = comm.bcast(check_error, root=0)
    if check_error is not None:
        raise RuntimeError(f"input check failed on rank 0 -- {check_error}")
    
    
   

    #env_id grew by two trailing integer fields -- '_<dmin_bin>_<dmax_bin>' in the
    #_dists token, '_<dmin_bin>_<a_bin>' in this one. Both worst-case at '_999_999'
    #= 8 extra characters, so 128 -> 136 covers _and_A unchanged. NOTE that pandas
    #RAISES if a value exceeds min_itemsize, it does not truncate, so this is
    #fail-loud rather than silent.
    min_itemsize={'env_id': 136,
                  'mat_id' : 128, 
                  'central_element' : 16, 
                  'coord_symbol' : 128, 
                  'coord_elements' : 128,
                  'isite' : 8,
                  'omat_subset' : 128,
                  'omat_subset_idx' : 16
                  }

    #Prepare HDF5

    #
    #if rank < residual_n_samples:
    #else:
    #    start_sample = residual_n_samples * (avg_local_n_sample + 1) + (rank - residual_n_samples) * avg_local_n_sample

    ##If this is a restart calculation, shift start_sample to the last structure not yet written to file
    #if os.path.exists(local_hdf5_filename):
    #    with pd.HDFStore(local_hdf5_filename, mode='r') as store:

    #        if last_written_sample != end_sample - 1:
    #else:

    


    #with h5py.File(local_hdf5_filename, 'w') as f:


    global lgf
    lgf = LocalGeometryFinder()
    #structure_refinement="none" is ESSENTIAL: with the default ("refined"),
    #LocalGeometryFinder.setup_structure() internally replaces the structure with
    #SpacegroupAnalyzer(...).get_refined_structure() (the conventional standard
    #cell). All site indices in the returned LightStructureEnvironments then refer
    #to that internal cell, NOT to the structure we passed - mis-assigning
    #environments for centered (I/F/C) Bravais lattices and crashing on raw cells
    #whose refined cell has fewer atoms. With "none" chemenv uses exactly the cell
    #we hand it, keeping site indices aligned 1:1 with our atoms and magmoms.
    #Side effect: the continuous symmetry measure (csm -> perturb_scale) now
    #reflects the actual, non-idealised geometry of each structure.
    lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True,
                         structure_refinement=LocalGeometryFinder.STRUCTURE_REFINEMENT_NONE)
    print(f"LocalGeometryFinder has been initialized")

    #Collect all structure indices in a list
    structures = [s for s in range(n_samples)]
    print(f'Structures array has been initialized')


    n_computed_structures = 0
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            futures = [executor.submit(process_single_structure, s, dataset_path, min_itemsize) for s in structures]

            results = [future.result() for future in futures]

            n_computed_structures += len(results)
            print(f"Total number of computed structures: {n_computed_structures}")



    print("Classification of data set finished")

if __name__ == "__main__":
    main()
