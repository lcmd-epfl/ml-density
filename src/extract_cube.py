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
from libs.variance_lib import compute_molecule_sigma

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


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description='Export a predicted density, a prediction-minus-reference difference, or a full-covariance PITC predictive standard-deviation field as a .cube file.')
    parser.add_argument('-c', '--config', default='config.txt', help='Path to config.txt used by the project. Relative paths inside it are resolved from the working directory, so run this script from the directory holding config.txt.')
    parser.add_argument('--mts', help='Input metatensor file containing predicted density coefficients. Defaults to the prediction file the config points to for this --training subset and the last training fraction of the config. Ignored with --std.')
    parser.add_argument('-m', '--mol', type=int, default=0, help='Local molecule index inside the test/training set (position in the set, not the row in the xyz file).')
    parser.add_argument('-t', '--training', action='store_true', help='--mol indexes the training set instead of the test set.')
    parser.add_argument('-o', '--output', help='Output .cube file path. Defaults to <field>_<mol>.cube, where <field> is density, diff or std depending on the flags.')
    parser.add_argument('-g', '--grid', type=int, default=80, help='Cube grid size in each dimension.')
    parser.add_argument('-s', '--std', action='store_true', help='Output the predictive standard-deviation field sigma[rho(r)] = sqrt(phi(r)^T Sigma_c* phi(r)) instead of the mean density. Carries the same units as the density (e/bohr^3), so it can be read against rho(r) on a shared scale, unlike the variance (e^2/bohr^6). Requires full_gpr=True in the config and ignores --mts.')
    parser.add_argument('-d', '--diff', action='store_true', help='Output the error field rho_pred(r) - rho_ref(r) instead of the predicted mean density, using the ab initio coefficients stored by preprocess.py.')
    parser.add_argument('--chunk', type=int, default=4096, help='Grid points processed per chunk in --std mode, to bound memory (the ao_values @ Sigma_c* intermediate is O(chunk * nao^2)).')
    args = parser.parse_args()
    if args.std and args.diff:
        parser.error('--diff and --std are mutually exclusive')
    if not args.output:
        field = 'std' if args.std else 'diff' if args.diff else 'density'
        args.output = f'{field}_{args.mol}.cube'
    return args


def main():  # noqa: D103
    args = parse_args()
    o, p = read_config(config_path=args.config)
    frac = o.fracs[-1]

    subsets = Subset(p.train_test_sets)
    configs, pred_path = subsets.get_pred_idx(args.training, p.predictions, frac)
    xyz_index = int(configs[args.mol])

    atoms = ase.io.read(p.xyzfilename, ':')[xyz_index]
    atomic_numbers = atoms.get_atomic_numbers()
    mol = make_pyscf_mol(atomic_numbers, atoms.get_positions(), basis=o.basisname, ignore=True)

    cube = pyscf.tools.cubegen.Cube(mol, nx=args.grid, ny=args.grid, nz=args.grid)
    coords = cube.get_coords()

    if args.std:
        if not o.full_gpr:
            msg = '--std requires full_gpr=True in the config (no PITC Cholesky factor exists otherwise)'
            raise RuntimeError(msg)

        ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
        basis = Basis(o.basisname, elements=ref_elements)
        l_factor = np.load(p.cholesky_pitc.format(train_frac=frac))
        _, l_mm = kmm_cholesky(basis, ref_elements, metatensor.load(p.kernel_mm), o.jit)

        sigma_star = compute_molecule_sigma(basis, atomic_numbers, xyz_index, ref_elements,
                                             l_factor, l_mm, p.kernel_nm, p.power_spectrum)
        sigma_star = reorder.reorder_ao(mol, sigma_star, src='gpr', dest='pyscf')
        if sigma_star.shape[0]!=mol.nao_nr():
            msg = f'Covariance dimension {sigma_star.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)

        field_values = evaluate_std_field(mol, coords, sigma_star, args.chunk)
    else:
        mts_path = args.mts or pred_path
        if not os.path.exists(mts_path):
            msg = (f'{mts_path} does not exist -- run prediction.py for this subset and fraction, '
                   'pass --mts explicitly, or check that this script runs where config.txt lives')
            raise RuntimeError(msg)
        pred = load_coeff_tensor(mts_path, args.mol)
        # predictions are saved baselined (per-element average subtracted at training time);
        # add it back to get the actual density coefficients
        averages = metatensor.load(p.spherical_averages)
        tmap_add(pred, averages)
        coeffs = tensormap_to_array(mol, pred, dest='gpr', fast=True)
        if args.diff:
            c0 = np.load(p.clean_coefficients.format(xyz_index))
            if c0.shape!=coeffs.shape:
                msg = f'Reference coefficient vector length {c0.shape[0]} does not match prediction length {coeffs.shape[0]}'
                raise ValueError(msg)
            coeffs -= c0
        coeffs = reorder.reorder_ao(mol, coeffs, src='gpr', dest='pyscf')
        if coeffs.shape[0]!=mol.nao_nr():
            msg = f'Coefficient vector length {coeffs.shape[0]} does not match PySCF AO count {mol.nao_nr()}'
            raise ValueError(msg)

        field_values = mol.eval_ao('GTOval_sph', coords) @ coeffs

    cube.write(field_values.reshape(args.grid, args.grid, args.grid), args.output)


if __name__=='__main__':
    main()
