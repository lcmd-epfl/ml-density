#!/usr/bin/env python3
"""Solve the regression system and save model weights."""

import gc
import numpy as np
import pandas as pd
import scipy.linalg as spl
from numba import jit
import metatensor
from libs.tmap import vector2tmap
from libs.config import get_settings
from libs.functions import Basis
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()

    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, ref_elements)
    totsize = basis.nao_for_mol(ref_elements)

    k_MM = metatensor.load(p.kernel_mm)
    mat  = np.ndarray((totsize,totsize))
    idx = basis.sparse_indices(ref_elements)

    logger.debug(f'problem dimensionality = {totsize}')

    for frac in o.fracs:
        Avec = np.loadtxt(p.avec.format(train_frac=frac))
        mat[:] = 0
        fill_matrix(mat, k_MM, p.bmat.format(train_frac=frac), idx, basis.nmax, o.jit, o.reg)
        weights = spl.solve(mat, Avec, assume_a='sym', lower=True, overwrite_a=True, overwrite_b=True)
        weights = vector2tmap(ref_elements, basis, weights)
        metatensor.save(p.weights.format(train_frac=frac), weights)


@jit(nopython=True)
def unravel_tril(mat, data, jitter):
    """Fill a lower-triangular matrix from its vector form and add a constant to the diagonal.

    One should not use `np.tril_indices()` because it creates a huge indices array.

    Args:
        mat (np.ndarray): Updated in place.
        data (np.ndarray): Packed lower-triangular coefficients.
        jitter (float): Diagonal jitter added for numerical stability.
    """
    n = mat.shape[0]
    k = 0
    for j in range(n):
        for i in range(j+1):
            mat[j,i] = data[k]
            k += 1
        mat[j,j] += jitter


def fill_matrix(mat, k_MM, bmatfile, idx, nmax, jitter, reg):
    """Assemble the regression matrix from the B matrix and add regularization.

    Args:
        mat (np.ndarray): Vector representation of the lower triangle of a symmetric matrix.
        k_MM (metatensor.TensorMap): Reference-reference kernel.
        bmatfile (str): Path to packed B-matrix binary file.
        idx (np.ndarray): Sparse AO start indices per reference and l.
        nmax (dict[int, np.ndarray]): Radial basis sizes indexed by element and l.
        jitter (float): Diagonal jitter added after unpacking B.
        reg (float): Regularization coefficient scaling kernel blocks.
    """
    data = np.fromfile(bmatfile)
    unravel_tril(mat, data, jitter)
    del data
    gc.collect()
    for (l, q), kblock in k_MM.items():
        msize = 2*l+1
        for iiref12, (iref1, iref2) in enumerate(kblock.samples):
            if iref1<iref2:
                continue
            dk = reg * kblock.values[iiref12,:,:,0]
            for n in range(nmax[q][l]):
                i1 = idx[iref1, l] + n*msize
                i2 = idx[iref2, l] + n*msize
                mat[i1:i1+msize, i2:i2+msize] += dk


if __name__=='__main__':
    main()
