#!/usr/bin/env python3
"""Compute prediction errors and electron number diagnostics."""

import os
import logging
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


def correct_number_of_electrons(c, metric, q, N):
    """Project coefficients onto the subspace with the constrained electron number.

    Args:
        c (np.ndarray[float]): Input coefficient vector.
        metric (np.ndarray[float]): Metric matrix.
        q (np.ndarray[float]): Number of electrons in each AO.
        N (int | float): Target number of electrons.

    Returns:
        np.ndarray: Corrected coefficient vector with q @ c equal to N.
    """
    metric_inv_q = np.linalg.solve(metric, q)
    return c + metric_inv_q * (N - c@q)/(q@metric_inv_q)


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


def table_legend(use_charges, *, has_variance):
    """Build the legend explaining the columns of the printed error table.

    Args:
        use_charges (str | None): Mode controlling which dataset column is used (None, "charge", or "N").
        has_variance (bool): Whether predicted-variance columns were printed for at least one fraction.

    Returns:
        str: Legend text; the electron-number-correction and variance lines are included only when relevant.
    """
    return '\n'.join([
        '',
        'TABLE LEGEND',
        'mol # i (j)  : molecule i within this test set (dataset index j)',
        'baselined    : (c - c0)^T M (c - c0) / (c0 - c_av)^T M (c0 - c_av) * 100',
        'relative     : (c - c0)^T M (c - c0) / c0^T J c0 * 100',
        'absolute     : (c - c0)^T M (c - c0)',
        *(['pred var     : Tr(Sigma_c* M) / (c - c_av)^T M (c - c_av) * 100'] if has_variance else []),
        'nel_pred     : predicted number of electrons, q^T c',
        'nel_ref      : reference number of electrons, q^T c0',
        'ΔN           : nel_pred - nel_ref for this molecule',
        *(['corr N       : baselined relative error after projecting the prediction onto q^T c = N.'] if use_charges else []),
        'MAE          : mean of the above over all molecules in the fraction',
        'MAX          : max of the above over all molecules in the fraction',
        'frac         : fraction of the training set actually used to fit the model being evaluated',
        *(['corr(baselined, pred var): Pearson correlation between "baselined" and "pred var" across the fraction\'s molecules'] if has_variance else []),
        'where',
        'c    = predicted coefficients',
        'c0   = reference (ab initio) coefficients',
        'c_av = per-element average coefficients, subtracted before training and added back at prediction time',
        'M    = Metric used. Currently it is the 2-center Coulomb (RI) metric, int int phi_i(r) phi_j(r\')/|r-r\'| dr dr\' based on '
        'Briling, K. R., Fabrizio, A. & Corminboeuf, C. Impact of quantum-chemical metrics on the machine learning prediction of electron density. J. Chem. Phys. 155, 024107 (2021).',
        'q    = number of electrons per atomic orbital',
        '',
        'Notes:',
        'The baselined error isolates the ML-predicted part (c - c_av) from the trivial part (c_av),',
        'making it the most meaningful measure of model error.',
        *(['',
           'pred var is the a priori counterpart of the baselined error: Sigma_c* is the GPR predictive covariance',
           'over c - c0, so Tr(Sigma_c* M) = E[(c - c0)^T M (c - c0)] estimates "absolute" before c0 is known.',
           'Its magnitude is NOT calibrated: the kernel carries no amplitude hyperparameter, so the prior sits far',
           'above the true scale of the density and pred var reads orders of magnitude high. Rank molecules by it',
           '-- which is what corr(baselined, pred var) measures -- but do not read its absolute value.'] if has_variance else []),
        '',
        ])


def load_variances(o, p, training, frac):
    """Load per-molecule whole-molecule predictive variance produced by variance.py, if available.

    Args:
        o (types.SimpleNamespace): Options namespace.
        p (types.SimpleNamespace): Paths namespace.
        training (bool): Whether the training or test subset is being reported (matches variance.py's --training).
        frac (float): Training fraction.

    Returns:
        dict[int, float] | None: Mapping from dataset molecule index to the dimensionless
        predicted variance relative to the baselined density, as a percentage, or None if
        full_gpr is disabled or variance.py hasn't been run yet for this subset/fraction.
    """
    if not o.full_gpr:
        return None
    path = p.var_trace.format(subset='training' if training else 'test', train_frac=frac)
    if not os.path.exists(path):
        logger.info(f'full_gpr=True but {path} does not exist -- run variance.py to include predicted variance here')
        return None
    var_df = pd.read_csv(path)
    return dict(zip(var_df['mol_idx'], var_df['var_relative'], strict=True))


def evaluate_molecule(o, p, df, atomic_numbers, averages, norms, N_all, pred, imol, itest, npred, pred_var):
    """Compute one molecule's prediction error and format its report line.

    Args:
        o (types.SimpleNamespace): Options namespace.
        p (types.SimpleNamespace): Paths namespace.
        df (pd.DataFrame): Dataset CSV, for the molecule's xyz file name.
        atomic_numbers (list[np.ndarray]): Atomic numbers for each dataset molecule.
        averages (metatensor.TensorMap): Per-element average coefficients.
        norms (np.ndarray): Per-molecule (norm, norm_baselined) pairs.
        N_all (np.ndarray): Per-molecule target electron counts.
        pred (metatensor.TensorMap): This molecule's predicted (baselined) coefficients.
        imol (int): Dataset index of the molecule.
        itest (int): Position of the molecule within the current subset.
        npred (int): Number of molecules in the current subset.
        pred_var (float | None): Predicted whole-molecule variance from variance.py, if available.

    Returns:
        tuple[Error, list[str]]: The molecule's Error, and its formatted column values in table
        order (mol label, baselined, relative, absolute, [pred var], nel_pred, nel_ref, ΔN,
        [corr N], xyz file). Column widths aren't decided here -- that needs every molecule's
        fields first, so it's the caller's job (see format_row).
    """
    atoms  = atomic_numbers[imol]
    N      = N_all[imol]
    mol    = make_dummy_mol(atoms=atoms, basis=o.basisname, charge=sum(atoms)-N, spin=N%2)
    qvec   = rho_moments(mol, rho=None, moments=(0,), per_atom=False)[0]
    metric = tensormap_to_array(mol, metatensor.load(p.metric_matrix.format(imol)), dest='gpr', fast=True)
    c0     = np.load(p.clean_coefficients.format(imol))
    norm, norm_bl = norms[imol]

    tmap_add(pred, averages)
    c = tensormap_to_array(mol, pred, dest='gpr', fast=True)
    dc = c - c0

    error = Error()
    error.abs    = dc @ metric @ dc
    error.rel    = error.abs/norm * 100.0
    error.rel_bl = error.abs/norm_bl * 100.0

    N_c0 = qvec @ c0
    N_pred = qvec @ c
    dN = N_pred - N_c0  # signed, so the printed row reads exactly as nel_pred - nel_ref = ΔN
    errorn_rel_bl = None
    if o.use_charges:
        error.N = abs(dN)
        dcn = correct_number_of_electrons(c, metric, qvec, N) - c0
        errorn_rel_bl = (dcn @ metric @ dcn) / norm_bl * 100.0

    fields = [
        f'mol # {itest:{len(str(npred))}} ({imol:{len(str(len(atomic_numbers)))}}):',
        f'{error.rel_bl:.2e} %',
        f'{error.rel:.2e} %',
        f'{error.abs:.2e}',
        ]
    if pred_var is not None:
        fields.append(f'{pred_var:.2e} %')
    fields += [f'{N_pred:.4f}', f'{N_c0:.4f}', f'{dN:+.4f}']
    if o.use_charges:
        fields.append(f'{errorn_rel_bl:.2e} %')
    fields.append(p.xyz.format(mol_name=df['id'][imol]))
    return error, fields


def format_row(fields, widths, separators):
    """Center-pad fields to per-column widths and join them with per-gap separators.

    Args:
        fields (list[str]): Formatted field values, in column order.
        widths (list[int]): Column width for every field except the last.
        separators (list[str]): Separator placed after each field except the last (see
            main()'s `gap_after`: normally '   ', but ' - ' and ' = ' around nel_pred/nel_ref/ΔN
            so the row reads as the equation nel_pred - nel_ref = ΔN).

    Returns:
        str: The assembled, aligned row.
    """
    padded = (f'{field:^{width}}' for field, width in zip(fields, widths, strict=True))
    return ''.join(f'{field}{sep}' for field, sep in zip(padded, separators, strict=True))


def summary_line(label, err, headers, widths, separators, *, use_charges, var_value=None):
    """Format one MAE/MAX-style summary line, aligned to the table columns above it.

    Args:
        label (str): Line label, e.g. 'MAE' or 'MAX'.
        err (Error): Aggregated (mean- or max-reduced) error for baselined, relative and absolute error.
        headers (list[str]): Table column headers, from main().
        widths (list[int]): Table column widths, from main().
        separators (list[str]): Table column separators, from main().
        use_charges (str | None): Mode controlling which dataset column is used (None, "charge", or "N").
        var_value (float | None): Aggregated predicted variance to report, if available.

    Returns:
        str: The formatted summary line.
    """
    fields = ['' for _ in headers]
    # - 2 because = should be aligned with :
    fields[0] = f'{label:>{widths[0] - 2}} ='
    fields[headers.index('baselined')] = f'{err.rel_bl:.2e} %'
    fields[headers.index('relative')] = f'{err.rel:.2e} %'
    fields[headers.index('absolute')] = f'{err.abs:.2e}'
    if var_value is not None:
        fields[headers.index('pred var')] = f'{var_value:.2e} %'
    if use_charges:
        fields[headers.index('ΔN')] = f'{err.N:+.4f}'
    blanks = [' ' * len(sep) for sep in separators]
    return format_row(fields, widths, blanks).rstrip()


def get_settings_quiet(return_args):
    """Load settings while suppressing get_settings' INFO "Configuration file: ..." log line.

    That line is emitted by config parsing during get_settings, so it can only be silenced around
    the call itself. INFO (and below) is disabled just for this call, keeping this script's stdout
    to the report table alone (it is redirected to a file); WARNING/ERROR still get through, and
    other scripts are unaffected.

    Args:
        return_args (list[str]): Script-specific CLI flags to parse (forwarded to get_settings).

    Returns:
        tuple: (args, options, paths), as returned by get_settings.
    """
    logging.disable(logging.INFO)
    try:
        return get_settings(return_args=return_args)
    finally:
        logging.disable(logging.NOTSET)


def main():  # noqa: D103
    args, o, p = get_settings_quiet(return_args=['training'])

    df = pd.read_csv(p.dataset)
    subsets = Subset(p.train_test_sets)
    averages = metatensor.load(p.spherical_averages)
    atomic_numbers = moldata_read(p.xyzfilename)
    N_all = get_number_of_electrons(o.use_charges, atomic_numbers, df)
    norms = np.load(p.coef_norms)

    any_variance = False
    for frac in o.fracs:
        pred_configs, pred_path = subsets.get_pred_idx(args.training, p.predictions, frac)
        predictions = split(metatensor.load(pred_path))
        npred = len(pred_configs)
        variances = load_variances(o, p, args.training, frac)
        any_variance |= variances is not None

        headers = ['', 'baselined', 'relative', 'absolute']
        if variances is not None:
            headers.append('pred var')
        headers += ['nel pred', 'nel ref', 'ΔN']
        if o.use_charges:
            headers.append('corr N')
        headers.append('xyz file')

        # nel_pred - nel_ref = ΔN, spelled out with real operators instead of the usual 3-space gap
        gap_after = {'nel pred': '  -  ', 'nel ref': '  =  '}
        separators = [gap_after.get(headers[i], '   ') for i in range(len(headers))]

        total = Error()
        max_error = Error()
        # pred var is E[dc^T M dc] over the same baselined norm and the same metric, so "baselined"
        # is its like-for-like partner: this correlation is a calibration check, not a comparison
        # across metrics.
        bl_errors, pred_vars, rows = [], [], []
        for itest, imol in enumerate(pred_configs):
            pred_var = variances[imol] if variances is not None else None
            error, fields = evaluate_molecule(o, p, df, atomic_numbers, averages, norms, N_all,
                                              predictions[itest], imol, itest, npred, pred_var)
            total += error
            for key in Error.fields:
                setattr(max_error, key, max(getattr(max_error, key), getattr(error, key), key=abs))
            if pred_var is not None:
                bl_errors.append(error.rel_bl)
                pred_vars.append(pred_var)
            rows.append(fields)

        widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
        print(format_row(headers, widths, separators))
        for fields in rows:
            print(format_row(fields, widths, separators))

        total /= npred
        mean_var = np.mean(pred_vars) if variances is not None else None
        max_var = max(pred_vars) if variances is not None else None
        corr_extra = ''
        if variances is not None:
            corr = np.corrcoef(bl_errors, pred_vars)[0, 1] if npred > 1 else float('nan')
            corr_extra = f'   corr(baselined, pred var) = {corr:.2f}'
        print()
        print(summary_line('MAE', total, headers, widths, separators, use_charges=o.use_charges, var_value=mean_var))
        print(summary_line('MAX', max_error, headers, widths, separators, use_charges=o.use_charges, var_value=max_var))
        print(f"frac={frac}", corr_extra)

    print(table_legend(o.use_charges, has_variance=any_variance), end='')


if __name__=='__main__':
    main()
