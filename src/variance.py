#!/usr/bin/env python3
"""Run PITC whole-molecule predictive-variance computation for train/test/extrapolation subsets.

Reports one number per molecule, computed a priori without the reference coefficients c0:

    var_relative = Tr(Sigma_c* J) / [(c - c_av)^T J (c - c_av)] * 100

Both numerator and denominator are Coulomb-metric and carry units of Hartree, so the ratio is
dimensionless; the *100 matches compute_error.py's convention for its own error columns, which
likewise scale in the code and print a bare ' %'.

The denominator is the *baselined* density rho - rho_av, not rho itself, because this column
exists to be the a priori counterpart of compute_error.py's "baselined" column. That one reports
dc^T J dc / (c0 - c_av)^T J (c0 - c_av), and Tr(Sigma_c* J) = E[dc^T J dc], so dividing by the
same baselined norm is what puts the two on one scale and makes the correlation between them
meaningful. Only the predicted c is used, never c0: extrapolation molecules have no reference.

Normalising by the full c^T J c instead would break that correspondence, and would add a size
bias on top. Measured over this example's 8 test molecules (nel 36-84, so the fits are indicative
rather than converged): the un-baselined norm is core-dominated (~nel^1.70, corr(nel) = 1.00,
with the per-element averages carrying ~98% of it) while Tr(Sigma_c* J) is essentially size-free
on its own (~nel^0.00, corr(nel) = 0.03), so dividing by it drives the ratio to ~nel^-1.71
(corr -0.88) against the baselined norm's ~nel^-0.94 (corr -0.69).

See libs/variance_lib.py for why the Coulomb metric is the one that makes this number comparable
to the error it is meant to predict.
"""

import os
from functools import partial
import numpy as np
import pandas as pd
import ase.io
import metatensor
from tqdm import tqdm
from qstack import reorder
from libs.config import get_settings
from libs.functions import Basis, Subset, get_dataset_paths, make_dummy_mol, remove_averages
from libs.pitc_lib import kmm_cholesky
from libs.tmap import tmap2averages
from libs.variance_lib import compute_molecule_sigma, molecule_variance_trace, compute_coulomb_metric, density_self_repulsion
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def coeff_path_formatter(args, o, p, frac):
    """Build the path formatter for prediction.py's exported coefficient files.

    Mirrors prediction.py::_get_split, which is what writes these files: the extrapolation
    subset has no train_frac in its path and saves no joined .mts (pred_path is None there),
    so the per-molecule .dat files are the one source that covers every subset uniformly.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (only .extra is used).
        o (types.SimpleNamespace): Options namespace.
        p (types.SimpleNamespace): Paths namespace.
        frac (float): Training fraction.

    Returns:
        Callable[..., str]: Formatter taking imol= and returning the coefficient file path.
    """
    if args.extra:
        return partial(p.extra_predicted_coeff.format, order=o.output_coeff_order)
    return partial(p.predicted_coeff.format, train_frac=frac, order=o.output_coeff_order)


def load_predicted_coeffs(c_fmter, o, atoms, imol):
    """Load one molecule's predicted density coefficients, in GPR AO order.

    Args:
        c_fmter (Callable[..., str]): Formatter from coeff_path_formatter.
        o (types.SimpleNamespace): Options namespace.
        atoms (np.ndarray[int]): Atomic numbers of the molecule.
        imol (int): Dataset index of the molecule.

    Returns:
        np.ndarray: Density coefficient vector, GPR AO order.

    Raises:
        RuntimeError: If prediction.py has not exported this molecule's coefficients yet.
    """
    path = c_fmter(imol=imol)
    if not os.path.exists(path):
        msg = (f'{path} does not exist -- run prediction.py for this subset first. The '
               'dimensionless var_relative column normalises by int (rho(r) - rho_av(r))^2 dr '
               'of the predicted density, which is what prediction.py exports.')
        raise RuntimeError(msg)
    coeffs = np.loadtxt(path)
    if o.output_coeff_order!='gpr':
        # positions do not affect AO ordering, so a dummy mol is enough to reorder
        mol = make_dummy_mol(atoms, basis=o.basisname, ignore=True)
        coeffs = reorder.reorder_ao(mol, coeffs, dest='gpr', src=o.output_coeff_order)
    return coeffs


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['training', 'extra'])

    if not o.full_gpr:
        msg = 'variance.py requires full_gpr=True in the config (no PITC Cholesky factor exists otherwise)'
        raise RuntimeError(msg)

    dataset_paths = get_dataset_paths(p, extra=args.extra)
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=ref_elements)
    av_coefs = tmap2averages(metatensor.load(p.spherical_averages))
    _, l_mm = kmm_cholesky(basis, ref_elements, metatensor.load(p.kernel_mm), o.jit)
    frac_list = o.fracs[-1:] if args.extra else o.fracs

    # every subset needs real geometry: the Coulomb metric behind Tr(Sigma_c* J) is computed from
    # it rather than loaded, so extrapolation molecules take the same path (see variance_lib).
    asemols = ase.io.read(dataset_paths.xyz, ':')
    atomic_numbers = [mol.get_atomic_numbers() for mol in asemols]
    positions = [mol.get_positions() for mol in asemols]

    if args.extra:
        test_configs = np.arange(len(atomic_numbers))
        subset = 'extra'
    else:
        subsets = Subset(p.train_test_sets)
        subset = 'training' if args.training else 'test'

    for frac in frac_list:
        l_factor = np.load(p.cholesky_pitc.format(train_frac=frac))

        if not args.extra:
            test_configs = subsets.get_training(frac) if args.training else subsets.get_test()

        logger.debug(f'Number of {subset} molecules = {len(test_configs)}')

        c_fmter = coeff_path_formatter(args, o, p, frac)

        traces = []
        for imol in tqdm(test_configs):
            atoms = atomic_numbers[imol]
            metric = compute_coulomb_metric(atoms, positions[imol], o.basisname)
            sigma_star = compute_molecule_sigma(basis, atoms, imol, ref_elements, l_factor, l_mm,
                                                 dataset_paths.kernel, dataset_paths.power)
            var_coul = molecule_variance_trace(sigma_star, metric)
            coeffs = load_predicted_coeffs(c_fmter, o, atoms, imol)
            rho_bl_coul = density_self_repulsion(remove_averages(basis.index(atoms), coeffs, av_coefs), metric)
            traces.append((imol, var_coul/rho_bl_coul * 100.0))

        pd.DataFrame(traces, columns=['mol_idx', 'var_relative']).to_csv(
            p.var_trace.format(subset=subset, train_frac=frac), index=False)


if __name__=='__main__':
    main()
