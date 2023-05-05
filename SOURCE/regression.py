#!/usr/bin/env python3

import sys
import gc
import numpy as np
import scipy.linalg as spl
from basis import basis_read
from config import read_config
from functions import moldata_read, get_elements_list, nao_for_mol
from libs.tmap import sparseindices_fill
from libs.get_matrices_B import mpos
import equistore


def main():
    o, p = read_config(sys.argv)

    _, _, atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)
    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    ref_elements = np.hstack(atomic_numbers)[ref_indices]

    (el_dict, lmax, nmax) = basis_read(p.basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    k_MM = equistore.load(f'{p.kmmbase}{o.M}.npz')
    totsize = nao_for_mol(ref_elements, lmax, nmax)
    mat  = np.ndarray((totsize,totsize))
    idx = sparseindices_fill(lmax, nmax, ref_elements)

    print(f'problem dimensionality = {totsize}')

    for frac in o.fracs:
        avecfile    = f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt'
        bmatfile    = f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat'
        weightsfile = f'{p.weightsfilebase}_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
        Avec = np.loadtxt(avecfile)
        mat[:] = 0

        fill_matrix(mat, k_MM, bmatfile, idx, nmax, o.jit, o.reg)

        weights = spl.solve(mat, Avec, assume_a='sym', lower=True, overwrite_a=True, overwrite_b=True)
        np.save(weightsfile, weights)


def fill_matrix(mat, k_MM, bmatfile, idx, nmax, jit, reg):
    n = mat.shape[0]
    data = np.fromfile(bmatfile)
    # shouldn't use np.tril_indices because it creates huge indices arrays
    k = 0
    for j in range(n):
        for i in range(j+1):
            mat[j,i] = data[mpos(i,j)]
            k += 1
        mat[j,j] += jit
    del data
    gc.collect()

    for (l, q), kblock in k_MM:
        msize = 2*l+1
        for iiref12, (iref1, iref2) in enumerate(kblock.samples):
            if iref1<iref2:
                continue
            dk = reg * kblock.values[iiref12,:,:,0]
            for n in range(nmax[q, l]):
               i1 = idx[iref1, l] + n*msize
               i2 = idx[iref2, l] + n*msize
               mat[i1:i1+msize, i2:i2+msize] += dk


if __name__=='__main__':
    main()
