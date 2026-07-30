#!/usr/bin/env python3
"""Export a predicted density, its error against the ab initio density, or the predictive standard-deviation field, as a .cube file."""

import argparse
import os
import ase.io
import metatensor
import numpy as np
import pandas as pd
import pyscf.tools
from qstack import reorder
from qstack.io.metatensor import split, tensormap_to_array
from libs.config import read_config
from libs.functions import Basis, Subset, make_pyscf_mol
from libs.tmap import tmap_add
from libs.logger_setup import setup_logger
from libs.pitc_lib import kmm_cholesky
from libs.variance_lib import compute_molecule_sigma, load_sigma_f2

logger = setup_logger(__name__, __file__)


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
            paths = [args.output + f'{args.mol}_std.cube' if args.std else None,
                      args.output + f'{args.mol}_ref.cube' if args.ref else None,
                      args.output + f'{args.mol}_pred.cube' if args.pred else None,
                      args.output + f'{args.mol}_diff.cube' if args.diff else None]
            paths = [path for path in paths if path is not None]
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
                        logger.error(f'All {len(configs)} molecules have already been extracted for ALL asked field ({', '.join(asked_fields)}). Exiting.')
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
    parser.add_argument('-s', '--std', action='store_true', help='Output only the predictive standard-deviation field sigma[rho(r)] = sqrt(phi(r)^T Sigma_c* phi(r)) instead of the mean density. Carries the same units as the density (e/bohr^3). Requires full_gpr=True in the config and ignores --mts.')
    parser.add_argument('-o', '--output', default="CUBE/",help='Prefix for output .cube files. File name constructed by <prefix><field>_<mol>.cube, where <field> is ref, pred, diff or std depending on the flags (<prefix> default is CUBE/).')
    parser.add_argument('-m', '--mol', type=int, help='Local molecule index inside the test/training set (position in the set, not the row in the xyz file) (default: a random density-unextracted molecule).')
    parser.add_argument('-t', '--training', action='store_true', help='--mol indexes the training set instead of the test set.')
    parser.add_argument('--mts', help='Input metatensor file containing predicted density coefficients. Defaults to the prediction file the config points to for this --training subset and the last training fraction of the config. Ignored with --std.')
    parser.add_argument('-g', '--grid', type=int, default=80, help='Cube grid size in each dimension (default: 80).')
    parser.add_argument('--chunk', type=int, default=4096, help='Grid points processed per chunk in --std mode, to bound memory (the ao_values @ Sigma_c* intermediate is O(chunk * nao^2)) (default: 4096).')
    args = parser.parse_args()
    if not (args.std or args.diff or args.ref or args.pred):
        args.std = True
        args.ref = True
        args.pred = True
        args.diff = True
        logger.info('All output fields will be written.')
    return args


def main():  # noqa: D103
    args = parse_args()
    o, p = read_config(config_path=args.config)
    frac = o.fracs[-1]

    subsets = Subset(p.train_test_sets)
    configs, pred_path = subsets.get_pred_idx(args.training, p.predictions, frac)
    xyz_index = select_molecule_index(configs, args)

    atoms = ase.io.read(p.xyzfilename, ':')[xyz_index]
    atomic_numbers = atoms.get_atomic_numbers()
    mol = make_pyscf_mol(atomic_numbers, atoms.get_positions(), basis=o.basisname, ignore=True)

    cube = pyscf.tools.cubegen.Cube(mol, nx=args.grid, ny=args.grid, nz=args.grid)
    coords = cube.get_coords()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    # STD
    if args.std:
        if not o.full_gpr:
            msg = 'Standard-deviation field requires full_gpr=True in the config (no PITC Cholesky factor exists otherwise)'
            logger.info(msg)
        else:
            ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
            basis = Basis(o.basisname, elements=ref_elements)
            l_factor = np.load(p.cholesky_pitc.format(train_frac=frac))
            _, l_mm = kmm_cholesky(basis, ref_elements, metatensor.load(p.kernel_mm), o.jit)

            sigma_star = compute_molecule_sigma(basis, atomic_numbers, xyz_index, ref_elements,
                                                l_factor, l_mm, p.kernel_nm, p.power_spectrum,
                                                load_sigma_f2(p.sigma_f2.format(train_frac=frac)))
            sigma_star = reorder.reorder_ao(mol, sigma_star, src='gpr', dest='pyscf')
            if sigma_star.shape[0]!=mol.nao_nr():
                msg = f'Covariance dimension {sigma_star.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
                raise ValueError(msg)

            field_values_std = evaluate_std_field(mol, coords, sigma_star, args.chunk)
            cube.write(field_values_std.reshape(args.grid, args.grid, args.grid), args.output + f'{args.mol}_std.cube')

    # Load the good coefficients for reference, predicted or both (diff).
    if args.ref or args.diff:
        coeffs_ref = np.load(p.clean_coefficients.format(xyz_index))
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
        coeffs_pred = reorder.reorder_ao(mol, coeffs_pred, src='gpr', dest='pyscf')
        if coeffs_pred.shape[0]!=mol.nao_nr():
            msg = f'Coefficient vector length {coeffs_pred.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)

    if args.ref:
        field_values_ref = mol.eval_ao('GTOval_sph', coords) @ coeffs_ref
        cube.write(field_values_ref.reshape(args.grid, args.grid, args.grid), args.output + f'{args.mol}_ref.cube')
    if args.pred:
        field_values_pred = mol.eval_ao('GTOval_sph', coords) @ coeffs_pred
        cube.write(field_values_pred.reshape(args.grid, args.grid, args.grid), args.output + f'{args.mol}_pred.cube')
    if args.diff:
        field_values_diff = mol.eval_ao('GTOval_sph', coords) @ (coeffs_pred - coeffs_ref)
        cube.write(field_values_diff.reshape(args.grid, args.grid, args.grid), args.output + f'{args.mol}_diff.cube')

if __name__=='__main__':
    main()
