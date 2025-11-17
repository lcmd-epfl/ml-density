#!/usr/bin/env python3

import sys
import itertools
import numpy as np
from tqdm import tqdm
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, get_elements_list, nao_for_mol, Basis
from libs.tmap import averages2tmap, vector2tmap, matrix2tmap, tmap2matrix


def main():
    o, p = read_config(sys.argv)

    print(f'{o.copy_metric=}')
    print(f'{o.reorder_ao=}')

    atomic_numbers = moldata_read(p.xyzfilename)
    nenv = dict(zip(*get_elements_list(atomic_numbers, return_counts=True)))

    basis = Basis(o.basisname, elements=nenv.keys())

    coefficients = load_coefs(len(atomic_numbers), p.coefffilebase)

    av_coefs = get_averages(nenv, basis, coefficients, atomic_numbers)

    for imol, (coef, atoms) in tqdm([*enumerate(zip(coefficients, atomic_numbers))]):

        ao_index = basis.index(atoms)

        idx = reorder_idx(ao_index, o.reorder_ao)

        good_coef = coef[idx]
        np.save(f'{p.goodcoeffilebase}{imol}.npy', good_coef)
        good_coef = remove_averages(ao_index, good_coef, av_coefs)

        if o.copy_metric:
            over      = np.load(f'{p.overfilebase}{imol}.npy')
            good_over = over[np.ix_(idx,idx)]
            over_tmap = matrix2tmap(atoms, basis.lmax, basis.nmax, good_over)
            metatensor.save(f'{p.goodoverfilebase}{imol}.mts', over_tmap)
        else:
            good_over = tmap2matrix(atoms, basis.lmax, basis.nmax, metatensor.load(f'{p.goodoverfilebase}{imol}.mts'))

        proj = good_over @ good_coef
        proj_tmap = vector2tmap(atoms, basis.lmax, basis.nmax, proj)
        metatensor.save(f'{p.baselinedwbase}{imol}.mts', proj_tmap)
    metatensor.save(p.avfile, averages2tmap(av_coefs))


def load_coefs(n, coefffilebase):
    coefficients = []
    for imol in range(n):
        try:
            coef = np.loadtxt(f'{coefffilebase}{imol}.dat')
        except:
            coef = np.load(f'{coefffilebase}{imol}.npy')
        coefficients.append(coef)
    return coefficients


def remove_averages(ao_index, coef, av_coefs):
    coef_new = np.copy(coef)
    for iat, q in enumerate(ao_index.atoms):
        coef_new[ao_index.find(iat=iat, l=0)] -= av_coefs[q]
    return coef_new


def reorder_idx(ao_index, reorder_ao=False):
    idx = np.arange(ao_index.nao, dtype=int)
    if reorder_ao:
        for iat in range(ao_index.nat):
            ao1 = ao_index.find(iat=iat, l=1)
            if len(ao1)==0:
                continue
            idx[ao1] = idx[ao1+np.tile([1,1,-2], len(ao1)//3)]
    return idx




def get_averages(nenv, basis, coefficients, atomic_numbers):
    av_coefs = {q: np.zeros(basis.nmax[(q, 0)]) for q in nenv.keys()}

    for coef, atoms in zip(coefficients, atomic_numbers):
        ao_index = basis.index(atoms)
        for iat, q in enumerate(atoms):
            av_coefs[q] += coef[ao_index.find(iat=iat, l=0)]

    for q in av_coefs.keys():
        av_coefs[q] /= nenv[q]
    return av_coefs


if __name__=='__main__':
    main()
