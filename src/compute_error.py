#!/usr/bin/env python3
"""Compute prediction errors and electron number diagnostics."""

import numpy as np
import pandas as pd
import metatensor
from qstack.io.metatensor import split, tensormap_to_array
from qstack.fields.moments import r2_c as rho_moments
from libs.config import get_settings
from libs.functions import moldata_read, make_dummy_mol, Subset
from libs.tmap import tmap_add
from libs.logger_setup import setup_logger

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
    fields = ('abs', 'rel', 'rel_bl', 'N')

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


def table_legend(use_charges):
    """Build the legend explaining the columns of the printed error table.

    Args:
        use_charges (str | None): Mode controlling which dataset column is used (None, "charge", or "N").

    Returns:
        str: Legend text; the electron-number-correction lines are included only if use_charges is set.
    """
    return '\n'.join([
        '',
        'TABLE LEGEND',
        'mol # i (j)  : molecule i within this test set (dataset index j)',
        'baselined    : (c - c0)^T S (c - c0) / (c0 - c_av)^T S (c0 - c_av) * 100',
        'relative     : (c - c0)^T S (c - c0) / c0^T S c0 * 100',
        'absolute     : (c - c0)^T S (c - c0)',
        'nel_pred     : predicted number of electrons, q^T c',
        'nel_ref      : reference number of electrons, q^T c0',
        'N            : expected number of electrons (nuclear charge, or from the dataset charge/N column)',
        *(['corr N       : baselined relative error after projecting the prediction onto q^T c = N.'] if use_charges else []),
        'MAE          : mean of the above errors over all molecules in the fraction',
        *(['ΔN           : mean |nel_pred - N| over all molecules'] if use_charges else []),
        'where',
        'c    = predicted coefficients',
        'c0   = reference (ab initio) coefficients',
        'c_av = per-element average coefficients, subtracted before training and added back at prediction time',
        'S    = metric (overlap) matrix',
        'q    = number of electrons per atomic orbital',
        '',
        'The baselined error isolates the ML-predicted part (c - c_av) from the trivial part (c_av),',
        'making it the more meaningful measure of model error.',
        '',
        ])


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['training'])

    df = pd.read_csv(p.dataset)
    subsets = Subset(p.train_test_sets)
    averages = metatensor.load(p.spherical_averages)
    atomic_numbers = moldata_read(p.xyzfilename)
    N_all = get_number_of_electrons(o.use_charges, atomic_numbers, df)
    norms = np.load(p.coef_norms)

    for frac in o.fracs:
        logger.info(f'fraction = {frac}')

        pred_configs, pred_path = subsets.get_pred_idx(args.training, p.predictions, frac)
        predictions = split(metatensor.load(pred_path))
        npred = len(pred_configs)

        total = Error()

        print()
        indent = len(f'mol # {0:{len(str(npred))}} ({0:{len(str(len(atomic_numbers)))}}):  ') - 1
        print(f'{"":<{indent}}{"baselined":<10}   {"relative":^10}   ( {"absolute":^8} )   {"nel_pred":>8} / {"nel_ref":>8} ( {"N":^3} )')
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
                f'mol # {itest:{len(str(npred))}} ({imol:{len(str(len(atomic_numbers)))}}):   ',
                f'{error.rel_bl:8.2e} %  {error.rel:.2e} %  ( {error.abs:.2e} )   ',
                f'{N_pred:8.4f} / {N_c0:8.4f} ( {N:3d} )   ',
                f'(corr N: {errorn_rel_bl:8.2e} %)   ' if o.use_charges else '',
                f'{p.xyz.format(mol_name=df['id'][imol])}',
                ]))

        total /= npred
        print(''.join([
            f'\nfrac={frac}  MAE = {total.rel_bl:.2e} %  {total.rel:.2e} %  ( {total.abs:.2e} )',
            f'   ΔN: {total.N:.2e}' if o.use_charges else '',
            ]))

    print(table_legend(o.use_charges), end='')


if __name__=='__main__':
    main()
