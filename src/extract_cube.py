#!/usr/bin/env python3
"""Export a predicted density, its error against the ab initio density, or the predictive standard-deviation field, as a .cube file."""

import argparse
import logging
import os
import sys
from types import SimpleNamespace
import ase.io
import metatensor
import numpy as np
import pandas as pd
import pyscf.gto
import pyscf.tools
from ase.data import chemical_symbols
from qstack import reorder
from qstack.io.metatensor import split, tensormap_to_array
from libs.config import read_config
from libs.config_utils import defaults, GPR_DTC
from libs.functions import Basis, Subset, make_pyscf_mol
from libs.tmap import tmap_add
from libs.logger_setup import setup_logger
from libs.gp_common import kmm_cholesky
from libs.variance_lib import compute_molecule_sigma, prior_scale, dtc_lambda

logger = setup_logger(__name__, __file__)

L_LETTERS = 'spdfghi'


def load_coeff_tensors(mts_path, mol_indices):
    """Load a coefficient TensorMap once and select the block of each requested molecule.

    The file is read and split a single time, so exporting many molecules costs one load.

    Args:
        mts_path (str): Path to a metatensor file, either joined across molecules
            (sample name 'mol_id') or already a single molecule.
        mol_indices (list[int]): Local indices into the joined TensorMap. A file holding a single
            molecule carries no such index, and answers a single request only.

    Returns:
        list[metatensor.TensorMap]: One TensorMap per requested molecule, in the same order.

    Raises:
        ValueError: Several molecules are requested from a file holding only one.
    """
    tensor = metatensor.load(mts_path)
    if tensor.sample_names and tensor.sample_names[0]=='mol_id':
        tensors = split(tensor)
        return [tensors[mol_index] for mol_index in mol_indices]
    if len(mol_indices) > 1:
        msg = (f'{mts_path} holds a single molecule -- its samples carry no mol_id -- so it cannot '
               f'answer the {len(mol_indices)} molecules requested with --mol '
               f'{" ".join(map(str, mol_indices))}. Pass the joined prediction file that '
               'prediction.py writes, or ask for a single molecule.')
        raise ValueError(msg)
    return [tensor]


def evaluate_std_field(mol, coords, sigma_star, chunk):
    """Evaluate sigma[rho(r)] = sqrt(phi(r)^T Sigma_c* phi(r)) on a grid.

    Args:
        mol (pyscf.gto.Mole): Molecule defining the AO basis (PySCF AO order).
        coords (np.ndarray): Grid points, shape (ngrid, 3).
        sigma_star (np.ndarray): Dense (nao, nao) predictive covariance, PySCF AO order.
        chunk (int): Grid points per chunk, bounding the O(chunk * nao^2) intermediate.

    Returns:
        np.ndarray: Standard-deviation field, shape (ngrid,), non-negative.
    """
    variance = np.empty(coords.shape[0])
    for start in range(0, coords.shape[0], chunk):
        stop = min(start+chunk, coords.shape[0])
        ao_chunk = mol.eval_ao('GTOval_sph', coords[start:stop])
        variance[start:stop] = np.einsum('gi,gi->g', ao_chunk @ sigma_star, ao_chunk)

    # Sigma_c* is PSD in exact arithmetic, so Var[rho(r)] >= 0 and the sqrt is always real.
    # Roundoff still puts grid points marginally below 0, which would sqrt to NaN, so clip.
    # A value far below 0 instead means Sigma_c* itself lost PSD-ness and the field is not
    # trustworthy -- report it rather than hiding it under the clip.
    worst = variance.min()
    if worst < -1e-8*max(variance.max(), 1.0):
        logger.warning(f'Var[rho(r)] reaches {worst:.3e} on the grid, well below 0: Sigma_c* is not '
                       'numerically positive semi-definite (consider raising the config jitter). '
                       'Clipping to 0 before the sqrt, but treat this field with suspicion.')
    return np.sqrt(variance.clip(min=0.0))


def azimuthal_type(value):
    """Parse one --azimuthal value: an integer l or a spectroscopic letter.

    Args:
        value (str): Command-line token, e.g. '1' or 'p'.

    Returns:
        int: Angular momentum.

    Raises:
        argparse.ArgumentTypeError: The token is neither a non-negative integer nor a letter of L_LETTERS.
    """
    if value.lower() in L_LETTERS:
        return L_LETTERS.index(value.lower())
    try:
        l = int(value)
    except ValueError:
        msg = f'{value!r} is neither an integer nor one of {", ".join(L_LETTERS)}'
        raise argparse.ArgumentTypeError(msg) from None
    if l < 0:
        msg = f'angular momentum must be non-negative, got {l}'
        raise argparse.ArgumentTypeError(msg)
    return l


def atom_type(value):
    """Parse one --atom value: a chemical symbol or a 1-based atom index.

    Args:
        value (str): Command-line token, e.g. 'C', 'o' or '15'.

    Returns:
        tuple[str, int]: ('q', atomic number) for a symbol, ('iat', 0-based index) for an index.

    Raises:
        argparse.ArgumentTypeError: The token is neither a known element symbol nor a positive integer.
    """
    try:
        index = int(value)
    except ValueError:
        symbol = value.capitalize()
        if symbol not in chemical_symbols[1:]:
            msg = f'{value!r} is neither an element symbol nor a positive atom index'
            raise argparse.ArgumentTypeError(msg) from None
        return ('q', chemical_symbols.index(symbol))
    if index < 1:
        msg = f'atom indices follow the xyz file and start at one, got {index}'
        raise argparse.ArgumentTypeError(msg)
    return ('iat', index-1)


def atom_label(entry):
    """Render one --atom entry the way it was written on the command line.

    Args:
        entry (tuple[str, int]): An ('q', atomic number) or ('iat', 0-based index) pair.

    Returns:
        str: Element symbol, or the 1-based atom index.
    """
    kind, value = entry
    return chemical_symbols[value] if kind=='q' else str(value+1)


def resolve_atoms(atomic_numbers, selection):
    """Resolve --atom entries into the 0-based indices of the atoms they select.

    Symbols and indices combine as a union: "-a O 15" is every oxygen plus atom 15.

    Args:
        atomic_numbers (np.ndarray[int]): Atomic numbers of the molecule, in xyz order.
        selection (list[tuple[str, int]] | None): Parsed --atom entries, None to keep every atom.

    Returns:
        np.ndarray[int] | None: Selected atom indices, None when no atom filter is set.

    Raises:
        ValueError: An index exceeds the number of atoms in this molecule.
    """
    if selection is None:
        return None
    keep = np.zeros(len(atomic_numbers), dtype=bool)
    for entry in selection:
        kind, value = entry
        if kind=='q':
            matched = atomic_numbers==value
            if not matched.any():
                logger.warning(f'This molecule ({make_formula(atomic_numbers)}) has no {atom_label(entry)} atom, '
                               'so that part of --atom selects nothing.')
            keep |= matched
        elif value >= len(atomic_numbers):
            msg = (f'--atom {atom_label(entry)} is out of range: this molecule ({make_formula(atomic_numbers)}) '
                   f'has {len(atomic_numbers)} atoms, indexed 1 to {len(atomic_numbers)} in xyz order')
            raise ValueError(msg)
        else:
            keep[value] = True
    return np.where(keep)[0]


def make_formula(atomic_numbers):
    """Build the chemical formula of a molecule, for error and log messages.

    Args:
        atomic_numbers (np.ndarray[int]): Atomic numbers of the molecule.

    Returns:
        str: Hill-ordered formula, e.g. 'C5H7NO'.
    """
    return ase.Atoms(numbers=atomic_numbers).get_chemical_formula()


def cube_path(args, mol_index, field):
    """Build the output .cube path for one field, encoding any (atom, l, n) selection.

    Layout: <output><mol>[_a<atom>-<atom>...][_l<l>-<l>...][_n<n>-<n>...][_tf<frac>]_<field>.cube,
    which reduces to the historical <output><mol>_<field>.cube when no selection flag is set and the
    default training fraction is exported.

    Args:
        args (argparse.Namespace): Parsed CLI arguments, carrying the args.frac_tag set by main().
        mol_index (int): Local molecule index, the <mol> part of the file name.
        field (str): One of 'std', 'ref', 'pred', 'diff'.

    Returns:
        str: Path of the .cube file holding that field.
    """
    tag = ''
    if args.atom is not None:
        tag += '_a' + '-'.join(map(atom_label, args.atom))
    if args.azimuthal is not None:
        tag += '_l' + '-'.join(map(str, args.azimuthal))
    if args.radial_channel is not None:
        tag += '_n' + '-'.join(map(str, args.radial_channel))
    return f'{args.output}{mol_index}{tag}{args.frac_tag}_{field}.cube'


def select_ao_mask(basis, atomic_numbers, args):
    """Build the boolean AO mask selecting the requested atoms and (l, n) channels, in GPR AO order.

    The three filters intersect: an AO survives when its atom, its l and its n are all selected.
    The mask is constant over the 2l+1 magnetic components of each shell, so it commutes with the
    gpr->pyscf reordering; it is nevertheless applied in GPR order, where the labels live.

    Args:
        basis (Basis): Basis covering the molecule's elements.
        atomic_numbers (np.ndarray[int]): Atomic numbers of the molecule, in xyz order.
        args (argparse.Namespace): Parsed CLI arguments (args.atom, args.azimuthal, args.radial_channel).

    Returns:
        np.ndarray[bool]: Mask of length nao, all True when no selection flag is set.

    Raises:
        ValueError: The requested selection matches no basis function at all.
    """
    ao_index = basis.index(atomic_numbers)
    iats = resolve_atoms(atomic_numbers, args.atom)
    keep = np.zeros(ao_index.nao, dtype=bool)
    keep[ao_index.find(iat=iats, l=args.azimuthal, n=args.radial_channel)] = True
    if not keep.any():
        available = ', '.join(f'{chemical_symbols[q]}: nmax={basis.nmax[q].tolist()}' for q in basis.elements.tolist())
        requested = ', '.join(f'{name}={value}' for name, value in
                              (('atoms', args.atom and list(map(atom_label, args.atom))),
                               ('l', args.azimuthal), ('n', args.radial_channel)) if value is not None)
        msg = (f'No basis function of {basis.basisname} matches the requested selection ({requested}). '
               f'The molecule is {make_formula(atomic_numbers)} and its radial channels per angular '
               f'momentum are -- {available}')
        raise ValueError(msg)
    if any(value is not None for value in (args.atom, args.azimuthal, args.radial_channel)):
        log_selection(basis, ao_index, iats, args, keep)
    return keep


def shell_exponents(basisname, q, nmax):
    """Collect the Gaussian exponent of each (l, n) radial channel of one element.

    Only meant for logging, so anything that does not map one-to-one onto the (l, n) channels of
    the Basis object -- a contracted or generally contracted auxiliary basis -- yields no exponent
    rather than a wrong one.

    Args:
        basisname (str): Basis-set name understood by PySCF.
        q (int): Atomic number.
        nmax (np.ndarray[int]): Number of radial channels per angular momentum, from Basis.nmax.

    Returns:
        dict[tuple[int, int], float]: Exponent keyed by (l, n), empty if the basis is contracted.
    """
    shells = {}
    for shell in pyscf.gto.basis.load(basisname, chemical_symbols[q]):
        l, primitives = shell[0], shell[1:]
        try:
            (exponent, _coefficient), = primitives
        except (TypeError, ValueError):  # contracted or generally contracted shell: no single exponent
            return {}
        shells.setdefault(l, []).append(exponent)
    if any(len(shells.get(l, []))!=n for l, n in enumerate(nmax)):
        return {}
    return {(l, n): exponent for l, exponents in shells.items() for n, exponent in enumerate(exponents)}


def log_selection(basis, ao_index, iats, args, keep):
    """Report which atoms and which shells the selection keeps, with their Gaussian exponents.

    The per-shell detail is only printed when an l or n filter is active: listing every shell of
    every element would otherwise bury the atom selection under a full dump of the basis.

    Args:
        basis (Basis): Basis covering the molecule's elements.
        ao_index (AOIndex): AO index of the molecule.
        iats (np.ndarray[int] | None): Selected atom indices, None when every atom is kept.
        args (argparse.Namespace): Parsed CLI arguments (args.azimuthal, args.radial_channel).
        keep (np.ndarray[bool]): The AO mask being reported.
    """
    atoms = ao_index.atoms
    if iats is not None:
        listed = ', '.join(f'{iat+1} ({chemical_symbols[atoms[iat]]})' for iat in iats)
        logger.info(f'Restricting the exported field to {len(iats)}/{len(atoms)} atoms of '
                    f'{make_formula(atoms)}, numbered as in the xyz file: {listed}')
    if args.azimuthal is not None or args.radial_channel is not None:
        for q in sorted({atoms[iat] for iat in (range(len(atoms)) if iats is None else iats)}):
            exponents = shell_exponents(basis.basisname, q, basis.nmax[q])
            shells = [(l, n) for l, nmax_l in enumerate(basis.nmax[q]) for n in range(nmax_l)
                      if (args.azimuthal is None or l in args.azimuthal)
                      and (args.radial_channel is None or n in args.radial_channel)]
            kept = sum(2*l+1 for l, _ in shells)
            selection = ', '.join(f'l={l} n={n}' + (f' (alpha={exponents[l, n]:.4g})' if exponents else '')
                                  for l, n in shells) or 'none'
            logger.info(f'Restricting the exported field to {kept}/{basis.nao_atom[q]} AOs per '
                        f'{chemical_symbols[q]} atom: {selection}')
    logger.info(f'Keeping {keep.sum()}/{keep.size} AOs of the molecule. This is a partial density: '
                'it does not integrate to the electron count, and is signed for l>0.')


def select_molecule_index(configs: list, args: argparse.Namespace) -> int:
    """Randomly select a molecule which has its density unextracted.

    Args:
        configs (list): List of molecule indices in the test/training set.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        int: Local index into the configs (position in the test/training set, not the row in the xyz file).
    """
    available_molecules = set(range(len(configs)))
    expended_selection = False
    while True:
        mol_index = int(np.random.choice(list(available_molecules)))
        paths = [cube_path(args, mol_index, field) for field in ('std', 'ref', 'pred', 'diff') if getattr(args, field)]
        missing = [path for path in paths if not os.path.exists(path)]
        # first pass: only molecules with no extracted field at all; after expansion: any missing field
        if len(missing)==len(paths) or (expended_selection and missing):
            logger.info(f'Randomly selected molecule {mol_index}/{len(configs)} for density extraction.')
            return mol_index
        available_molecules.remove(mol_index)
        if not available_molecules:
            asked_fields = {f for f in ["std", "ref", "pred", "diff"] if getattr(args, f)}
            if not expended_selection and len(asked_fields) > 1:
                logger.info(
                    f"All {len(configs)} molecules have already been extracted for at least one requested field ({', '.join(asked_fields)}). "
                    f"Expanding selection to molecules with at least one missing field among {', '.join(asked_fields)}."
                )
                available_molecules = set(range(len(configs)))
                expended_selection = True
            else:
                if len(asked_fields) == 1:
                    logger.error(f'All {len(configs)} molecules have already been extracted for the requested field ({", ".join(asked_fields)}). Exiting.')
                else:
                    logger.error(f'All {len(configs)} molecules have already been extracted for ALL asked field ({", ".join(asked_fields)}). Exiting.')
                sys.exit(1)


def resolve_molecules(configs, args):
    """Resolve the requested molecules into (local index, xyz row) pairs.

    Args:
        configs (list): List of molecule indices in the test/training set.
        args (argparse.Namespace): Parsed command-line arguments (args.mol, None to draw one at random).

    Returns:
        list[tuple[int, int]]: For each requested molecule, its local index in the test/training
            set and the row it occupies in the xyz file.

    Raises:
        ValueError: A requested index falls outside the test/training set.
    """
    if args.mol is None:
        mol_indices = [select_molecule_index(configs, args)]
    else:
        mol_indices = args.mol
        outside = [mol_index for mol_index in mol_indices if not 0 <= mol_index < len(configs)]
        if outside:
            subset = 'training' if args.training else 'test'
            msg = (f'--mol {" ".join(map(str, outside))} out of range: the {subset} set of this config '
                   f'holds {len(configs)} molecules, indexed 0 to {len(configs)-1}')
            raise ValueError(msg)
    return [(mol_index, int(configs[mol_index])) for mol_index in mol_indices]


def select_train_frac(o, args):
    """Pick the training fraction whose model is exported.

    Every per-fraction artefact -- the predictions, the Cholesky factor, the prior scale and the
    DTC lambda -- is read for this one fraction, so the exported fields always describe the same
    model. Without --train-frac that is the last fraction of the config, as before.

    Args:
        o (types.SimpleNamespace): Options namespace (reads o.fracs).
        args (argparse.Namespace): Parsed CLI arguments (args.train_frac, None for the default).

    Returns:
        float: The fraction as spelled in the config, so that it formats the very path templates
            regression.py and prediction.py wrote.

    Raises:
        ValueError: The requested fraction is not one of the config's train_fractions.
    """
    if args.train_frac is None:
        return o.fracs[-1]
    matches = np.isclose(o.fracs, args.train_frac)
    if not matches.any():
        msg = (f'--train-frac {args.train_frac} is not one of the train_fractions of {args.config}: '
               f'{", ".join(map(str, o.fracs.tolist()))}')
        raise ValueError(msg)
    return o.fracs[matches.argmax()]


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description='Export a predicted density, a reference, a prediction-minus-reference difference, or a PITC predictive standard-deviation field as a .cube file.')
    parser.add_argument('-c', '--config', default='config.txt', help='Path to config.txt used by the project. Relative paths inside it are resolved from the working directory, so run this script from the directory holding config.txt.')
    parser.add_argument('-p', '--pred', action='store_true', help='Output only the predicted density rho_pred(r).')
    parser.add_argument('-r', '--ref', action='store_true', help='Output only the ab-initio density rho_ref(r).')
    parser.add_argument('-d', '--diff', action='store_true', help='Output only the error field rho_pred(r) - rho_ref(r).')
    parser.add_argument('-s', '--std', action='store_true', help='Output only the predictive standard-deviation field sigma[rho(r)] = sqrt(phi(r)^T Sigma_c* phi(r)) instead of the mean density. Carries the same units as the density (e/bohr^3). Requires regression_model = gpr_DTC or gpr_PITC in the config and ignores --mts.')
    parser.add_argument('-o', '--output', default="CUBE/",help='Prefix for output .cube files. File name constructed by <prefix><mol>[_a<atom>...][_l<l>...][_n<n>...]_<field>.cube, where <field> is ref, pred, diff or std depending on the flags, the optional a/l/n parts record an --atom/--azimuthal/--radial-channel selection and the optional _tf<frac> part records a non-default --train-frac (<prefix> default is CUBE/, extended to CUBE/<tag> when the config is named config_<tag>.txt).')
    parser.add_argument('-a', '--atom', type=atom_type, nargs='+', help='Restrict the exported field to these atoms, given as element symbols (every atom of that element) or as 1-based indices into the molecule\'s block of the xyz file (that one atom). Symbols and indices may be mixed and are combined as a union, so "-a O 15" keeps every oxygen plus atom 15. Default: every atom.')
    parser.add_argument('-l', '--azimuthal', type=azimuthal_type, nargs='+', help='Restrict the exported field to these angular momenta, given as integers or spectroscopic letters (e.g. "-l 0 1" or "-l s p"). Default: every l of the basis.')
    parser.add_argument('-n', '--radial-channel', type=int, nargs='+', help='Restrict the exported field to these radial channels within each selected l. 0-based, ordered from the tightest to the most diffuse primitive -- the JKFIT auxiliary basis is uncontracted and has no principal quantum number. The number of channels differs per element and per l (cc-pvdz-jkfit: 10 s channels on C, 4 on H). Default: every radial channel.')
    parser.add_argument('-m', '--mol', type=int, nargs='+', help='Local molecule indices inside the test/training set (position in the set, not the row in the xyz file). Several molecules may be given at once, e.g. "-m 83 405", and are exported one after the other into their own .cube files (default: a single random density-unextracted molecule).')
    parser.add_argument('-t', '--training', action='store_true', help='--mol indexes the training set instead of the test set.')
    parser.add_argument('-f', '--train-frac', type=float, help='Export the model trained on this fraction of the training set, which must be one of the train_fractions of the config. It selects the prediction file, the Cholesky factor, the prior scale and (gpr_DTC) lambda together, so --pred and --std always describe the same model. A fraction other than the default is recorded in the output file name as _tf<frac> (default: the last training fraction of the config).')
    parser.add_argument('--mts', help='Input metatensor file containing predicted density coefficients. Defaults to the prediction file the config points to for this --training subset and --train-frac. Ignored with --std, which always reads the config artefacts of --train-frac.')
    parser.add_argument('-g', '--grid', type=int, default=80, help='Cube grid size in each dimension (default: 80).')
    parser.add_argument('--chunk', type=int, default=4096, help='Grid points processed per chunk in --std mode, to bound memory (the ao_values @ Sigma_c* intermediate is O(chunk * nao^2)) (default: 4096).')
    parser.add_argument('--log', type=str, default=logging._levelToName[defaults.loglevel], choices=logging._nameToLevel.keys(), help='Logging level. DEBUG adds the traceback of any molecule that fails to be extracted.')
    args = parser.parse_args()
    logger.setLevel(args.log)
    # sort and deduplicate so that e.g. "-l 1 0" and "-l s p" name the same output file
    for attr in ('azimuthal', 'radial_channel'):
        values = getattr(args, attr)
        if values is not None:
            setattr(args, attr, sorted(set(values)))
    if args.atom is not None:  # elements first, by atomic number, then the individual atoms
        args.atom = sorted(set(args.atom), key=lambda entry: (entry[0]!='q', entry[1]))
    if args.mol is not None:  # deduplicate, in the order given
        args.mol = list(dict.fromkeys(args.mol))
    if args.config != "config.txt" and args.output == "CUBE/":
        # a config named config_<tag>.txt gets its own output prefix, so runs of two configs never
        # silently overwrite each other's cubes; anything else cannot be named apart, so ask.
        name = os.path.basename(args.config)
        tag = name[len("config_"):-len(".txt")] if name.startswith("config_") and name.endswith(".txt") else ""
        if tag:
            args.output += tag
            logger.info(f"Config file is not default but output name is: deriving the output prefix "
                        f"{args.output!r} from {name!r}.")
        else:
            logger.warning("Config file is not default but output name is. Are you sure you will not overwrite important files?")
            answer = input("Continue? [y/N] ")
            if answer.lower() != "y":
                logger.info("Aborting.")
                sys.exit(0)
    return args


def export_molecule(args, o, p, mol_index, xyz_index, atoms, pred_tensor, inputs):
    """Write every field requested on the command line for one molecule.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        o (types.SimpleNamespace): Options namespace.
        p (types.SimpleNamespace): Paths namespace.
        mol_index (int): Local index in the test/training set, naming the output files.
        xyz_index (int): Row of the molecule in the xyz file, keying its stored data.
        atoms (ase.Atoms): The molecule itself.
        pred_tensor (metatensor.TensorMap | None): Its predicted coefficients, baselined and
            modified in place; None when neither --pred nor --diff is requested.
        inputs (types.SimpleNamespace): Molecule-independent data loaded once by main() -- the
            spherical averages and, with --std, the ingredients of Sigma_c*.

    Raises:
        ValueError: The (atom, l, n) selection matches no basis function of this molecule, or a
            stored array disagrees with the molecule's AO count.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    mol = make_pyscf_mol(atomic_numbers, atoms.get_positions(), basis=o.basisname, ignore=True)

    # AO mask for the requested (l, n) channels, in GPR AO order -- all True if neither flag is set
    keep = select_ao_mask(Basis(o.basisname, elements=atomic_numbers), atomic_numbers, args)

    cube = pyscf.tools.cubegen.Cube(mol, nx=args.grid, ny=args.grid, nz=args.grid)
    coords = cube.get_coords()

    if args.std:
        sigma_star = compute_molecule_sigma(inputs.model, inputs.ref_basis, atomic_numbers, xyz_index,
                                            inputs.ref_elements, inputs.l_factor, inputs.l_mm,
                                            p.kernel_nm, p.power_spectrum, inputs.sigma_p2, inputs.lam)
        # Var[sum_{i in S} c_i phi_i(r)] = phi_S(r)^T Sigma_SS phi_S(r)
        sigma_star *= np.outer(keep, keep)
        sigma_star = reorder.reorder_ao(mol, sigma_star, src='gpr', dest='pyscf')
        if sigma_star.shape[0]!=mol.nao_nr():
            msg = f'Covariance dimension {sigma_star.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)

        field_values_std = evaluate_std_field(mol, coords, sigma_star, args.chunk)
        cube.write(field_values_std.reshape(args.grid, args.grid, args.grid), cube_path(args, mol_index, 'std'))

    # Load the good coefficients for reference, predicted or both (diff).
    if args.ref or args.diff:
        coeffs_ref = np.load(p.clean_coefficients.format(xyz_index))
        coeffs_ref = np.where(keep, coeffs_ref, 0.0)
        coeffs_ref = reorder.reorder_ao(mol, coeffs_ref, src='gpr', dest='pyscf')
        if coeffs_ref.shape[0]!=mol.nao_nr():
            msg = f'Coefficient vector length {coeffs_ref.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)
    if args.pred or args.diff:
        # predictions are saved baselined (per-element average subtracted at training time);
        # add it back to get the actual density coefficients
        tmap_add(pred_tensor, inputs.averages)
        coeffs_pred = tensormap_to_array(mol, pred_tensor, dest='gpr', fast=True)
        # masked after the average is added back, so an l>0 selection drops the l=0 average too
        coeffs_pred = np.where(keep, coeffs_pred, 0.0)
        coeffs_pred = reorder.reorder_ao(mol, coeffs_pred, src='gpr', dest='pyscf')
        if coeffs_pred.shape[0]!=mol.nao_nr():
            msg = f'Coefficient vector length {coeffs_pred.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)

    if args.ref:
        field_values_ref = mol.eval_ao('GTOval_sph', coords) @ coeffs_ref
        cube.write(field_values_ref.reshape(args.grid, args.grid, args.grid), cube_path(args, mol_index, 'ref'))
    if args.pred:
        field_values_pred = mol.eval_ao('GTOval_sph', coords) @ coeffs_pred
        cube.write(field_values_pred.reshape(args.grid, args.grid, args.grid), cube_path(args, mol_index, 'pred'))
    if args.diff:
        field_values_diff = mol.eval_ao('GTOval_sph', coords) @ (coeffs_pred - coeffs_ref)
        cube.write(field_values_diff.reshape(args.grid, args.grid, args.grid), cube_path(args, mol_index, 'diff'))


def main():  # noqa: D103
    args = parse_args()
    o, p = read_config(config_path=args.config)
    if not (args.std or args.diff or args.ref or args.pred):
        args.ref = True
        args.pred = True
        args.diff = True
        if o.is_gpr:
            args.std = True
        logger.info(f'All output fields ({", ".join([f for f in ["std", "ref", "pred", "diff"] if getattr(args, f)])}) will be written.')

    frac = select_train_frac(o, args)
    # the historical file names carry no fraction, so only a non-default one is tagged: default
    # runs keep their names, and two fractions of the same molecule never overwrite each other
    args.frac_tag = f'_tf{frac}' if frac!=o.fracs[-1] else ''

    subsets = Subset(p.train_test_sets)
    configs, pred_path = subsets.get_pred_idx(args.training, p.predictions, frac)
    molecules = resolve_molecules(configs, args)

    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname)

    # Everything from here to the loop is shared by every requested molecule, so it is read once.
    if args.std and not o.is_gpr:
        msg = 'Standard-deviation field requires regression_model = gpr_DTC or gpr_PITC in the config (no Cholesky factor exists otherwise)'
        logger.info(msg)
        args.std = False
    inputs = SimpleNamespace(averages=None, ref_elements=None, ref_basis=None, l_factor=None,
                             l_mm=None, sigma_p2=None, lam=None, model=o.regression_model)
    if args.std:
        inputs.ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
        inputs.ref_basis = Basis(o.basisname, elements=inputs.ref_elements)
        inputs.l_factor = np.load(p.cholesky.format(train_frac=frac))
        _, inputs.l_mm = kmm_cholesky(inputs.ref_basis, inputs.ref_elements, metatensor.load(p.kernel_mm), o.jit)
        inputs.sigma_p2 = prior_scale(o, p, frac)
        inputs.lam = dtc_lambda(o, p, frac) if o.regression_model==GPR_DTC else None
    pred_tensors = None
    if args.pred or args.diff:
        mts_path = args.mts or pred_path
        if not os.path.exists(mts_path):
            msg = (f'{mts_path} does not exist -- run prediction.py for this subset and training '
                   f'fraction {frac}, pass --mts explicitly, or check that this script runs where '
                   'config.txt lives')
            raise RuntimeError(msg)
        pred_tensors = load_coeff_tensors(mts_path, [mol_index for mol_index, _ in molecules])
        inputs.averages = metatensor.load(p.spherical_averages)

    xyz_frames = ase.io.read(p.xyzfilename, ':')

    failed = []
    for imol, (mol_index, xyz_index) in enumerate(molecules):
        if len(molecules) > 1:
            logger.info(f'Extracting molecule {mol_index} ({imol+1}/{len(molecules)}), '
                        f'row {xyz_index} of {p.xyzfilename}.')
        try:
            export_molecule(args, o, p, mol_index, xyz_index, xyz_frames[xyz_index],
                            pred_tensors[imol] if pred_tensors is not None else None, inputs)
        # One molecule can fail on its own -- no atom of the requested element, no stored
        # coefficients, an AO count disagreeing with the basis -- without that saying anything
        # about the molecules queued behind it, so report it and carry on to them. Any other
        # exception is a bug rather than a property of this molecule, and still stops the run.
        except (ValueError, OSError, KeyError) as error:
            logger.warning(f'Molecule {mol_index} (row {xyz_index} of {p.xyzfilename}) failed, so its '
                           f'remaining fields are skipped -- {type(error).__name__}: {error}')
            # the warning above quotes the exception, but not where it was raised: keep the
            # traceback that the pre-loop version printed, one --log DEBUG away
            logger.debug(f'Traceback of the molecule {mol_index} failure:', exc_info=True)
            failed.append(mol_index)

    if failed:
        logger.error(f'{len(failed)}/{len(molecules)} molecules were not fully extracted: '
                     f'{", ".join(map(str, failed))}')
        sys.exit(1)

if __name__=='__main__':
    main()
