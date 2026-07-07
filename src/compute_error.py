#!/usr/bin/env python3
"""Compute prediction errors and electron number diagnostics."""

import numpy as np
import pandas as pd
import metatensor
from qstack.io.metatensor import split, tensormap_to_array
from qstack.fields.moments import r2_c as rho_moments
from libs.config import get_settings
from libs.functions import moldata_read, Basis, make_dummy_mol, Subset
from libs.tmap import sph2vector
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def correct_number_of_electrons(c, S, q, N):
    """Project coefficients onto the subspace with the constrained electron number.

    Args:
        c (np.ndarray[float]): Input coefficient vector.
        S (np.ndarray[float]): Metric matrix.
        q (np.ndarray[float]): Number of electrons in each AO.
        N (float): Target number of electrons.

    Returns:
        np.ndarray: Corrected coefficient vector with q @ c equal to N.
    """
    S1q = np.linalg.solve(S, q)
    return c + S1q * (N - c@q)/(q@S1q)


def get_number_of_electrons(use_charges, atomic_numbers, df):
    """Compute the target number of electrons for each molecule.

    Args:
        use_charges (str | None): Mode controlling which dataset column is used (None, "charge", or "N").
        atomic_numbers (list[np.ndarray]): Atomic numbers for each molecule.
        df (pd.DataFrame): Dataset containing charge/electron-count columns.

    Returns:
        np.ndarray: Per-molecule electron counts used for evaluation.
    """
    if use_charges in {None, 'charge'}:
        nuc_charges = np.array([sum(atoms) for atoms in atomic_numbers])
        if use_charges is None:
            return nuc_charges
    inp = df[use_charges].to_numpy()
    if use_charges=='N':
        return inp
    return nuc_charges - inp  # use_charges=='charge'


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['training'])

    df = pd.read_csv(p.dataset)
    averages = metatensor.load(p.spherical_averages)
    atomic_numbers = moldata_read(p.xyzfilename)
    basis = Basis(o.basisname, elements=averages.keys.column('center_type'))
    N_all = get_number_of_electrons(o.use_charges, atomic_numbers, df)

    for frac in o.fracs:

        logger.info(f'fraction = {frac}')
        if args.training:
            test_configs = Subset(p.train_test_sets).get_training(frac)
            predictfile = p.predictions.format(subset='training', train_frac=frac)
        else:
            test_configs = Subset(p.train_test_sets).get_test()
            predictfile = p.predictions.format(subset='test', train_frac=frac)
        ntest = len(test_configs)
        predictions = split(metatensor.load(predictfile))

        total_N, total_abs, total_rel, total_rel_bl = 0.0, 0.0, 0.0, 0.0

        print()
        for itest, imol in enumerate(test_configs):

            atoms = atomic_numbers[imol]
            N = N_all[imol]
            mol = make_dummy_mol(atoms=atoms, basis=o.basisname, charge=sum(atoms)-N, spin=N%2)
            qvec = rho_moments(mol, rho=None, moments=(0,), per_atom=False)[0]

            S    = tensormap_to_array(mol, metatensor.load(p.metric_matrix.format(imol)), dest='gpr', fast=True)
            c0   = np.load(p.clean_coefficients.format(imol))
            c_bl = tensormap_to_array(mol, predictions[itest], dest='gpr', fast=True)
            c_av = sph2vector(atoms, basis, averages)

            c0_bl = c0 - c_av
            c   = c_bl + c_av
            dc  = c - c0

            norm     = c0    @ S @ c0
            norm_bl  = c0_bl @ S @ c0_bl

            total_abs    += (error        := dc @ S @ dc)
            total_rel    += (error_rel    := error/norm * 100.0)
            total_rel_bl += (error_rel_bl := error/norm_bl * 100.0)

            nel0 = qvec @ c0
            nel  = qvec @ c
            if o.use_charges:
                total_N += abs(nel - N)
                dcn = correct_number_of_electrons(c, S, qvec, N) - c0
                errorn_rel_bl = (dcn @ S @ dcn) / norm_bl * 100.0
            else:
                errorn_rel_bl = np.nan

            s1 = f'mol # {itest:{len(str(ntest))}} ({imol:{len(str(len(atomic_numbers)))}}):  '
            s2 = f'{error_rel_bl:8.3f} %  {error_rel:.2e} %    ( {error:.2e} )   {nel:8.4f} / {nel0:8.4f} ( {N:3d} )     (corr N: {errorn_rel_bl:8.3f} %)    {p.xyz.format(mol_name=df['id'][imol])}'
            print(s1+s2)

        print(f'\nfrac={frac}\tMAE = {total_rel_bl/ntest:.2e} %  {total_rel/ntest:.2e} %    ( {total_abs/ntest:.2e} )', end='')

        if o.use_charges:
            print(f'  ΔN: {total_N/ntest:.2e}')


if __name__=='__main__':
    main()
