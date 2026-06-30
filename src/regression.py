#!/usr/bin/env python3

import sys
import gc
import numpy as np
import pandas as pd
import scipy.linalg as spl
from numba import jit
import metatensor
from libs.config import read_config
from libs.functions import Basis


def main():
    o, p = read_config(sys.argv)

    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, np.unique(ref_elements))
    totsize = basis.nao_for_mol(ref_elements)

    k_MM = metatensor.load(p.kernel_mm)
    mat  = np.ndarray((totsize,totsize))
    idx = basis.sparse_indices(ref_elements)

    print(f'problem dimensionality = {totsize}')

    for frac in o.fracs:
        avecfile    = f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt'
        bmatfile    = f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat'
        weightsfile = p.weights.format(train_frac=frac)
        Avec = np.loadtxt(avecfile)
        mat[:] = 0

        fill_matrix(mat, k_MM, bmatfile, idx, basis.nmax, o.jit, o.reg)

        weights = spl.solve(mat, Avec, assume_a='sym', lower=True, overwrite_a=True, overwrite_b=True)
        np.save(weightsfile, weights)


@jit(nopython=True)
def unravel_tril(mat, data, jitter):
    # shouldn't use np.tril_indices() because it creates a huge indices array
    n = mat.shape[0]
    k = 0
    for j in range(n):
        for i in range(j+1):
            mat[j,i] = data[k]
            k += 1
        mat[j,j] += jitter
    return


def fill_matrix(mat, k_MM, bmatfile, idx, nmax, jitter, reg):
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
