#!/usr/bin/env python3
"""Compute prediction errors and electron number diagnostics."""

import numpy as np
import pandas as pd
import metatensor
from qstack.io.metatensor import split, tensormap_to_array
from qstack.fields.moments import r2_c as rho_moments
from libs.config import get_settings
from libs.functions import moldata_read, make_dummy_mol
from libs.tmap import tmap_add
from libs.logger_setup import setup_logger
from prediction import get_pred_idx

logger = setup_logger(__name__, __file__)


def correct_number_of_electrons(c, S, q, N):
    """Project coefficients onto the subspace with the constrained electron number.

    Args:
        c (np.ndarray[float]): Input coefficient vector.
        S (np.ndarray[float]): Metric matrix.
        q (np.ndarray[float]): Number of electrons in each AO.
        N (int | float): Target number of electrons.

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


class Error:
    """Prediction errors."""
    fields = ('abs','rel', 'rel_bl', 'N')

    def __init__(self):
        """Initialize an Error instance with 0."""
        for key in self.fields:
            setattr(self, key, 0.0)

    def __repr__(self):
        return self.__class__.__qualname__+'('+', '.join(f'{key}={getattr(self, key)}' for key in self.fields)+')'

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            for key in self.fields:
                setattr(self, key, getattr(self, key) + getattr(other, key))
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            for key in self.fields:
                setattr(self, key, getattr(self, key)/other)
            return self
        return NotImplemented


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['training'])

    df = pd.read_csv(p.dataset)
    averages = metatensor.load(p.spherical_averages)
    atomic_numbers = moldata_read(p.xyzfilename)
    N_all = get_number_of_electrons(o.use_charges, atomic_numbers, df)
    norms = np.load(p.coef_norms)

    for frac in o.fracs:
        logger.info(f'fraction = {frac}')

        pred_configs, pred_path = get_pred_idx(args, p, frac)
        predictions = split(metatensor.load(pred_path))
        npred = len(pred_configs)

        total = Error()

        print()
        for itest, imol in enumerate(pred_configs):

            atoms = atomic_numbers[imol]
            N    = N_all[imol]
            mol  = make_dummy_mol(atoms=atoms, basis=o.basisname, charge=sum(atoms)-N, spin=N%2)
            qvec = rho_moments(mol, rho=None, moments=(0,), per_atom=False)[0]
            S    = tensormap_to_array(mol, metatensor.load(p.metric_matrix.format(imol)), dest='gpr', fast=True)
            c0   = np.load(p.clean_coefficients.format(imol))
            norm, norm_bl = norms[imol]

            tmap_add(predictions[itest], averages)
            c = tensormap_to_array(mol, predictions[itest], dest='gpr', fast=True)
            dc = c - c0

            error = Error()
            error.abs    = dc @ S @ dc
            error.rel    = error.abs/norm * 100.0
            error.rel_bl = error.abs/norm_bl * 100.0

            N_c0 = qvec @ c0
            N_pred = qvec @ c
            if o.use_charges:
                error.N = abs(N_pred - N)
                dcn = correct_number_of_electrons(c, S, qvec, N) - c0
                errorn_rel_bl = (dcn @ S @ dcn) / norm_bl * 100.0

            total += error

            print(''.join([
                f'mol # {itest:{len(str(npred))}} ({imol:{len(str(len(atomic_numbers)))}}):  ',
                f'{error.rel_bl:8.3f} %  {error.rel:.2e} %    ( {error.abs:.2e} )   ',
                f'{N_pred:8.4f} / {N_c0:8.4f} ( {N:3d} )     ',
                f'(corr N: {errorn_rel_bl:8.3f} %)    ' if o.use_charges else '',
                f'{p.xyz.format(mol_name=df['id'][imol])}',
                ]))

        total /= npred
        print(''.join([
            f'\nfrac={frac}\tMAE = {total.rel_bl:.2e} %  {total.rel:.2e} %    ( {total.abs:.2e} )',
            '  ΔN: {total.N:.2e}' if o.use_charges else '',
            ]))


if __name__=='__main__':
    main()
