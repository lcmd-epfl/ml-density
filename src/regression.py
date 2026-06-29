#!/usr/bin/env python3
'''Solve the regularized KRR system (B + reg*K_MM + jit*I) x = A and save weights.'''

import sys
import gc
import numpy as np
import scipy.linalg as spl
from numba import jit
import metatensor
from libs.basis import basis_read
from libs.config import read_config
from libs.functions import nao_for_mol
from libs.tmap import sparseindices_fill


def main() -> None:
    o, p = read_config(sys.argv)

    _, lmax, nmax = basis_read(p.basisfilename)
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:, 1]
    totsize = nao_for_mol(ref_elements, lmax, nmax)

    k_MM = metatensor.load(f'{p.kmmbase}{o.M}.mts')
    mat = np.ndarray((totsize, totsize))
    idx = sparseindices_fill(lmax, nmax, ref_elements)
    # eye = np.eye(totsize)

    print(f'problem dimensionality = {totsize}')

    for frac in o.fracs:
        avecfile    = f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt'
        bmatfile    = f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat'
        weightsfile = f'{p.weightsfilebase}_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
        # covfile     = f'{p.weightsfilebase}_cov_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
        Avec = np.loadtxt(avecfile)
        mat[:] = 0

        fill_matrix(mat, k_MM, bmatfile, idx, nmax, o.jit, o.reg)

        weights = spl.solve(mat, Avec, assume_a='sym', lower=True, overwrite_a=True, overwrite_b=True)
        # chol = spl.cho_factor(mat, lower=True, overwrite_a=True, check_finite=False)
        # weights = spl.cho_solve(chol, Avec, overwrite_b=True, check_finite=False)
        # cov = spl.cho_solve(chol, eye, check_finite=False)
        np.save(weightsfile, weights)
        # np.save(covfile, cov)


@jit(nopython=True)
def unravel_tril(mat: np.ndarray, data: np.ndarray, jitter: float) -> None:
    '''Unpack a lower-triangular packed array into a matrix and add jitter to diagonal.'''
    n = mat.shape[0]
    k = 0
    for j in range(n):
        for i in range(j + 1):
            mat[j, i] = data[k]
            k += 1
        mat[j, j] += jitter
    return


def fill_matrix(mat: np.ndarray, k_MM: metatensor.TensorMap, bmatfile: str, idx: np.ndarray, nmax: dict, jitter: float, reg: float) -> None:
    '''
    Add jitter and regularization terms to matrix B
    B -> B + jit*I + reg*K_MM.
    '''
    data = np.fromfile(bmatfile)
    unravel_tril(mat, data, jitter)
    del data
    gc.collect()
    for (l, q), kblock in k_MM.items():
        msize = 2 * l + 1
        for iiref12, (iref1, iref2) in enumerate(kblock.samples):
            if iref1 < iref2:
                continue
            dk = reg * kblock.values[iiref12, :, :, 0]
            for n in range(nmax[q, l]):
                i1 = idx[iref1, l] + n * msize
                i2 = idx[iref2, l] + n * msize
                mat[i1:i1 + msize, i2:i2 + msize] += dk


if __name__ == '__main__':
    main()