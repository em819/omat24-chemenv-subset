#!/usr/bin/env python3
"""Find a sub-dataset's structures and decide whether they are an ASE database
or extxyz.

Used by both tokenize_omat24.py and check_input.py so the two cannot disagree.
Uses only the standard library, which keeps check_input.py runnable without
pymatgen, mpi4py, h5py or fairchem.
"""

import os

# Tried in order when looking for an extxyz file. OMAT_FILE_TEMPLATE replaces
# the whole filename instead.
EXTXYZ_SUFFIXES = ('.xyz', '.extxyz', '.xyz.gz', '.extxyz.gz')

# OMat24 sub-datasets whose frames carry no per-atom initial_magmoms (apart from
# a few aimd frames). ASE returns zeros for the missing array, so their <spin>
# field is 0, in the published tokens too. Every other sub-dataset, including
# any name not listed here, is expected to carry magmoms.
SUBSETS_WITHOUT_MAGMOMS = frozenset({
    'aimd-from-PBE-1000-npt', 'aimd-from-PBE-1000-nvt',
    'aimd-from-PBE-3000-npt', 'aimd-from-PBE-3000-nvt',
    'rattled-relax',
})


def magmoms_expected(subset):
    """True if frames of `subset` should carry initial_magmoms."""
    return subset not in SUBSETS_WITHOUT_MAGMOMS


def find_ase_db(db_dir, subset):
    """Return the ASE-DB directory for `subset`, or None."""
    path = os.path.join(db_dir, subset)
    return path if os.path.isdir(path) else None


def find_extxyz(db_dir, subset, template=None):
    """Return the extxyz file for `subset`, or None.

    `template` (default: $OMAT_FILE_TEMPLATE), e.g. 'all_structures_{subset}.xyz',
    replaces the filename entirely; otherwise the plain <subset><ext> forms are
    tried in the order of EXTXYZ_SUFFIXES.
    """
    if template is None:
        template = os.environ.get('OMAT_FILE_TEMPLATE')
    candidates = ([template.format(subset=subset)] if template else
                  [subset + ext for ext in EXTXYZ_SUFFIXES])
    for name in candidates:
        path = os.path.join(db_dir, name)
        if os.path.isfile(path):
            return path
    return None


def resolve_input(db_dir, subset, fmt='auto', template=None):
    """Return (path, 'ase-db'|'extxyz') for `subset`.

    fmt:
      'auto'   -- use whichever form exists. Raises if the sub-dataset is
                  available in NEITHER form, or in BOTH (the two can hold
                  different data, so the choice must be made deliberately).
      'ase-db' -- require db_dir/<subset>/
      'extxyz' -- require an extxyz file

    Raises FileNotFoundError / RuntimeError / ValueError with an actionable
    message; never returns a path that does not exist.
    """
    fmt = (fmt or 'auto').lower()

    if fmt == 'ase-db':
        path = find_ase_db(db_dir, subset)
        if path is None:
            raise FileNotFoundError(
                f"format 'ase-db' was requested but {os.path.join(db_dir, subset)} "
                f"is not a directory.")
        return path, 'ase-db'

    if fmt == 'extxyz':
        path = find_extxyz(db_dir, subset, template)
        if path is None:
            tried = ((template or os.environ.get('OMAT_FILE_TEMPLATE', '')).format(subset=subset)
                     or ', '.join(subset + e for e in EXTXYZ_SUFFIXES))
            raise FileNotFoundError(
                f"format 'extxyz' was requested but no extxyz for '{subset}' "
                f"under {db_dir} (tried: {tried}). Set OMAT_FILE_TEMPLATE if "
                f"your files are named differently.")
        return path, 'extxyz'

    if fmt != 'auto':
        raise ValueError(
            f"input format {fmt!r} is not supported; use 'auto', 'ase-db' or "
            f"'extxyz'.")

    db_path = find_ase_db(db_dir, subset)
    xyz_path = find_extxyz(db_dir, subset, template)
    if db_path and xyz_path:
        raise RuntimeError(
            f"'{subset}' is present BOTH as an ASE DB ({db_path}) and as extxyz "
            f"({xyz_path}). They may hold different data, so auto-detection "
            f"refuses to choose. Set OMAT_INPUT_FORMAT=ase-db or =extxyz.")
    if db_path:
        return db_path, 'ase-db'
    if xyz_path:
        return xyz_path, 'extxyz'
    raise FileNotFoundError(
        f"'{subset}' found under {db_dir} neither as a directory (ASE DB) nor "
        f"as {'/'.join(subset + e for e in EXTXYZ_SUFFIXES)} (extxyz). Check "
        f"OMAT_DATA_DIR, subsets.txt, and OMAT_FILE_TEMPLATE.")


def load_subsets(path):
    """Read subsets.txt: one sub-dataset name per line, '#' comments skipped."""
    with open(path) as f:
        names = [ln.strip() for ln in f]
    return [n for n in names if n and not n.startswith("#")]
