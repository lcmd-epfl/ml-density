#!/usr/bin/env python3

import sys
import gc
import numpy as np
import scipy.linalg as spl
from basis import basis_read
from config import read_config
from functions import moldata_read, get_elements_list, nao_for_mol
from libs.tmap import sparseindices_fill


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

    k_MM = np.load(f'{p.kmmbase}{o.M}.npy')
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

        fill_matrix(mat, k_MM, bmatfile, ref_elements, idx, lmax, nmax, o.jit, o.reg      )

        weights = spl.solve(mat, Avec, assume_a='sym', lower=True, overwrite_a=True, overwrite_b=True)
        np.save(weightsfile, weights)


def fill_matrix(mat, k_MM, bmatfile, ref_elements, idx, lmax, nmax, jit, reg):
    n = mat.shape[0]
    ind = np.tril_indices(n)
    data = np.fromfile(bmatfile)
    mat[ind] = data
    del ind
    del data
    gc.collect()

    mat[np.diag_indices(n)] += jit

    for q in set(ref_elements):
        qrefs = np.where(ref_elements==q)[0]
        for l in range(lmax[q]+1):
            msize = 2*l+1
            for iref1 in qrefs:
                for iref2 in qrefs:
                    dk = reg * k_MM[l][iref1*msize:(iref1+1)*msize, iref2*msize:(iref2+1)*msize]
                    for n in range(nmax[q, l]):
                       i1 = idx[iref1, l] + n*msize
                       i2 = idx[iref2, l] + n*msize
                       mat[i1:i1+msize, i2:i2+msize] += dk


if __name__=='__main__':
    main()
