#!/usr/bin/env python3
"""Export a predicted density, its error against the ab initio density, or the predictive standard-deviation field, as a .cube file."""

import argparse
import os
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
from libs.functions import Basis, Subset, make_pyscf_mol
from libs.tmap import tmap_add
from libs.logger_setup import setup_logger
from libs.pitc_lib import kmm_cholesky
from libs.variance_lib import compute_molecule_sigma, load_sigma_f2, variance_scale

logger = setup_logger(__name__, __file__)

L_LETTERS = 'spdfghi'


def load_coeff_tensor(mts_path, mol_index):
    """Load a coefficient TensorMap and select one molecule's block.

    Args:
        mts_path (str): Path to a metatensor file, either joined across molecules
            (sample name 'mol_id') or already a single molecule.
        mol_index (int): Local index into the joined TensorMap (ignored if not joined).

    Returns:
        metatensor.TensorMap: TensorMap for the selected molecule.
    """
    tensor = metatensor.load(mts_path)
    if tensor.sample_names and tensor.sample_names[0]=='mol_id':
        return split(tensor)[mol_index]
    return tensor


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


def cube_path(args, field):
    """Build the output .cube path for one field, encoding any (atom, l, n) selection.

    Layout: <output><mol>[_a<atom>-<atom>...][_l<l>-<l>...][_n<n>-<n>...]_<field>.cube, which
    reduces to the historical <output><mol>_<field>.cube when no selection flag is set.

    Args:
        args (argparse.Namespace): Parsed CLI arguments, with args.mol already resolved.
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
    return f'{args.output}{args.mol}{tag}_{field}.cube'


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
    """Select a molecule index. From args.mol if defined else randomly.
    
    Args:
        configs (list): List of molecule indices in the test/training set.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        int: Local index into the configs (position in the test/training set, not the row in the xyz file).
    """
    if args.mol is None:
        # Pick a random molecule which has its predicted density unextracted.
        import random
        mol_found = False
        available_molecules = set(range(len(configs)))
        expended_selection = False
        while not mol_found:
            args.mol = np.random.choice(list(available_molecules))
            paths = [cube_path(args, field) for field in ('std', 'ref', 'pred', 'diff') if getattr(args, field)]
            if not any(os.path.exists(path) for path in paths) if not expended_selection else not all(os.path.exists(path) for path in paths):
                logger.info(f'Randomly selected molecule {args.mol}/{len(configs)} for density extraction.')
                mol_found = True
            available_molecules.remove(args.mol)
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
                    import sys
                    sys.exit(1)
    xyz_index = int(configs[args.mol])
    return xyz_index

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
    parser.add_argument('-o', '--output', default="CUBE/",help='Prefix for output .cube files. File name constructed by <prefix><mol>[_a<atom>...][_l<l>...][_n<n>...]_<field>.cube, where <field> is ref, pred, diff or std depending on the flags and the optional a/l/n parts record an --atom/--azimuthal/--radial-channel selection (<prefix> default is CUBE/).')
    parser.add_argument('-a', '--atom', type=atom_type, nargs='+', help='Restrict the exported field to these atoms, given as element symbols (every atom of that element) or as 1-based indices into the molecule\'s block of the xyz file (that one atom). Symbols and indices may be mixed and are combined as a union, so "-a O 15" keeps every oxygen plus atom 15. Default: every atom.')
    parser.add_argument('-l', '--azimuthal', type=azimuthal_type, nargs='+', help='Restrict the exported field to these angular momenta, given as integers or spectroscopic letters (e.g. "-l 0 1" or "-l s p"). Default: every l of the basis.')
    parser.add_argument('-n', '--radial-channel', type=int, nargs='+', help='Restrict the exported field to these radial channels within each selected l. 0-based, ordered from the tightest to the most diffuse primitive -- the JKFIT auxiliary basis is uncontracted and has no principal quantum number. The number of channels differs per element and per l (cc-pvdz-jkfit: 10 s channels on C, 4 on H). Default: every radial channel.')
    parser.add_argument('-m', '--mol', type=int, help='Local molecule index inside the test/training set (position in the set, not the row in the xyz file) (default: a random density-unextracted molecule).')
    parser.add_argument('-t', '--training', action='store_true', help='--mol indexes the training set instead of the test set.')
    parser.add_argument('--mts', help='Input metatensor file containing predicted density coefficients. Defaults to the prediction file the config points to for this --training subset and the last training fraction of the config. Ignored with --std.')
    parser.add_argument('-g', '--grid', type=int, default=80, help='Cube grid size in each dimension (default: 80).')
    parser.add_argument('--chunk', type=int, default=4096, help='Grid points processed per chunk in --std mode, to bound memory (the ao_values @ Sigma_c* intermediate is O(chunk * nao^2)) (default: 4096).')
    args = parser.parse_args()
    # sort and deduplicate so that e.g. "-l 1 0" and "-l s p" name the same output file
    for attr in ('azimuthal', 'radial_channel'):
        values = getattr(args, attr)
        if values is not None:
            setattr(args, attr, sorted(set(values)))
    if args.atom is not None:  # elements first, by atomic number, then the individual atoms
        args.atom = sorted(set(args.atom), key=lambda entry: (entry[0]!='q', entry[1]))
    if args.config != "config.txt" and args.output == "CUBE/":
        logger.warning("Config file is not default but output name is. Are you sure you will not overwrite important files?")
    return args


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

    frac = o.fracs[-1]

    subsets = Subset(p.train_test_sets)
    configs, pred_path = subsets.get_pred_idx(args.training, p.predictions, frac)
    xyz_index = select_molecule_index(configs, args)

    atoms = ase.io.read(p.xyzfilename, ':')[xyz_index]
    atomic_numbers = atoms.get_atomic_numbers()
    mol = make_pyscf_mol(atomic_numbers, atoms.get_positions(), basis=o.basisname, ignore=True)

    # AO mask for the requested (l, n) channels, in GPR AO order -- all True if neither flag is set
    keep = select_ao_mask(Basis(o.basisname, elements=atomic_numbers), atomic_numbers, args)

    cube = pyscf.tools.cubegen.Cube(mol, nx=args.grid, ny=args.grid, nz=args.grid)
    coords = cube.get_coords()

    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname)

    # STD
    if args.std:
        if not o.is_gpr:
            msg = 'Standard-deviation field requires regression_model = gpr_DTC or gpr_PITC in the config (no Cholesky factor exists otherwise)'
            logger.info(msg)
        else:
            ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
            basis = Basis(o.basisname, elements=ref_elements)
            l_factor = np.load(p.cholesky.format(train_frac=frac))
            _, l_mm = kmm_cholesky(basis, ref_elements, metatensor.load(p.kernel_mm), o.jit)

            sigma_star = compute_molecule_sigma(basis, atomic_numbers, xyz_index, ref_elements,
                                                l_factor, l_mm, p.kernel_nm, p.power_spectrum,
                                                load_sigma_f2(p.sigma_f2.format(train_frac=frac)),
                                                variance_scale(o))
            # Var[sum_{i in S} c_i phi_i(r)] = phi_S(r)^T Sigma_SS phi_S(r)
            sigma_star *= np.outer(keep, keep)
            sigma_star = reorder.reorder_ao(mol, sigma_star, src='gpr', dest='pyscf')
            if sigma_star.shape[0]!=mol.nao_nr():
                msg = f'Covariance dimension {sigma_star.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
                raise ValueError(msg)

            field_values_std = evaluate_std_field(mol, coords, sigma_star, args.chunk)
            cube.write(field_values_std.reshape(args.grid, args.grid, args.grid), cube_path(args, 'std'))

    # Load the good coefficients for reference, predicted or both (diff).
    if args.ref or args.diff:
        coeffs_ref = np.load(p.clean_coefficients.format(xyz_index))
        coeffs_ref = np.where(keep, coeffs_ref, 0.0)
        coeffs_ref = reorder.reorder_ao(mol, coeffs_ref, src='gpr', dest='pyscf')
        if coeffs_ref.shape[0]!=mol.nao_nr():
            msg = f'Coefficient vector length {coeffs_ref.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)
    if args.pred or args.diff:
        mts_path = args.mts or pred_path
        if not os.path.exists(mts_path):
            msg = (f'{mts_path} does not exist -- run prediction.py for this subset and fraction, '
                   'pass --mts explicitly, or check that this script runs where config.txt lives')
            raise RuntimeError(msg)
        coeffs_pred = load_coeff_tensor(mts_path, args.mol)
        # predictions are saved baselined (per-element average subtracted at training time);
        # add it back to get the actual density coefficients
        averages = metatensor.load(p.spherical_averages)
        tmap_add(coeffs_pred, averages)
        coeffs_pred = tensormap_to_array(mol, coeffs_pred, dest='gpr', fast=True)
        # masked after the average is added back, so an l>0 selection drops the l=0 average too
        coeffs_pred = np.where(keep, coeffs_pred, 0.0)
        coeffs_pred = reorder.reorder_ao(mol, coeffs_pred, src='gpr', dest='pyscf')
        if coeffs_pred.shape[0]!=mol.nao_nr():
            msg = f'Coefficient vector length {coeffs_pred.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)

    if args.ref:
        field_values_ref = mol.eval_ao('GTOval_sph', coords) @ coeffs_ref
        cube.write(field_values_ref.reshape(args.grid, args.grid, args.grid), cube_path(args, 'ref'))
    if args.pred:
        field_values_pred = mol.eval_ao('GTOval_sph', coords) @ coeffs_pred
        cube.write(field_values_pred.reshape(args.grid, args.grid, args.grid), cube_path(args, 'pred'))
    if args.diff:
        field_values_diff = mol.eval_ao('GTOval_sph', coords) @ (coeffs_pred - coeffs_ref)
        cube.write(field_values_diff.reshape(args.grid, args.grid, args.grid), cube_path(args, 'diff'))

if __name__=='__main__':
    main()
