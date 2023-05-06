#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import read_config
from basis import basis_read
from functions import moldata_read, print_progress, get_elements_list, nao_for_mol
from libs.tmap import averages2tmap, vector2tmap, matrix2tmap, tmap2matrix


def main():
    o, p = read_config(sys.argv)

    print(f'{o.copy_metric=}')
    print(f'{o.reorder_ao=}')

    nmol, _, atomic_numbers = moldata_read(p.xyzfilename)
    lmax, nmax = basis_read(p.basisfilename)
    coefficients = load_coefs(atomic_numbers, p.coefffilebase)
    av_coefs = get_averages(lmax, nmax, coefficients, atomic_numbers)

    for imol, (coef, atoms) in enumerate(zip(coefficients, atomic_numbers)):
        print_progress(imol, nmol)
        idx = reorder_idx(atoms, lmax, nmax, o.reorder_ao)

        good_coef = coef[idx]
        np.save(f'{p.goodcoeffilebase}{imol}.npy', good_coef)
        good_coef = remove_averages(atoms, lmax, nmax, good_coef, av_coefs)

        if o.copy_metric:
            over      = np.load(f'{p.overfilebase}{imol}.npy')
            good_over = over[np.ix_(idx,idx)]
            over_tmap = matrix2tmap(atoms, lmax, nmax, good_over)
            equistore.save(f'{p.goodoverfilebase}{imol}.npz', over_tmap)
            np.save(f'{p.goodoverfilebase}{imol}.npy', good_over)
        else:
            good_over = tmap2matrix(atoms, lmax, nmax, equistore.load(f'{p.goodoverfilebase}{imol}.npz'))

        proj = good_over @ good_coef
        proj_tmap = vector2tmap(atoms, lmax, nmax, proj)
        equistore.save(f'{p.baselinedwbase}{imol}.npz', proj_tmap)
        np.savetxt(f'{p.baselinedwbase}{imol}.dat', proj)
    equistore.save(p.avfile, averages2tmap(av_coefs))


def load_coefs(atomic_numbers, coefffilebase):
    coefficients = []
    for imol, atoms in enumerate(atomic_numbers):
        try:
            coef = np.loadtxt(f'{coefffilebase}{imol}.dat')
        except:
            coef = np.load(f'{coefffilebase}{imol}.npy')
        coefficients.append(coef)
    return coefficients


def remove_averages(atoms, lmax, nmax, coef, av_coefs):
    coef_new = np.copy(coef)
    i = 0
    for q in atoms:
        coef_new[i:i+nmax[(q,0)]] -= av_coefs[q]
        for l in range(lmax[q]+1):
            i += (2*l+1)*nmax[(q,l)]
    return coef_new


def reorder_idx(atoms, lmax, nmax, reorder_ao=False):
    nao = nao_for_mol(atoms, lmax, nmax)
    idx = np.arange(nao, dtype=int)
    if reorder_ao:
        i = 0
        for q in atoms:
            i += nmax[(q,0)]
            if(lmax[q]<1):
                continue
            for n in range(nmax[(q,1)]):
                idx[i  ] = i+1
                idx[i+1] = i+2
                idx[i+2] = i
                i += 3
            for l in range(2, lmax[q]+1):
                i += (2*l+1)*nmax[(q,l)]
    return idx


def get_averages(lmax, nmax, coefficients, atomic_numbers):
    elements, counts = get_elements_list(atomic_numbers, return_counts=True)
    nenv = dict(zip(elements, counts))
    av_coefs = {q: np.zeros(nmax[(q, 0)]) for q in elements}
    for coef, atoms in zip(coefficients, atomic_numbers):
        i = 0
        for q in atoms:
            av_coefs[q] += coef[i:i+nmax[(q,0)]]
            for l in range(lmax[q]+1):
                i += (2*l+1)*nmax[(q,l)]
    for q in elements:
        av_coefs[q] /= nenv[q]
    return av_coefs


if __name__=='__main__':
    main()
