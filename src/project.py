#!/usr/bin/env python3

import sys
import numpy as np
from tqdm import tqdm
import metatensor
from qstack import compound, reorder, equio
from libs.config import read_config
from libs.functions import moldata_read, get_elements_list, Basis
from libs.tmap import averages2tmap


def main():
    o, p = read_config(sys.argv)

    print(f'{o.copy_metric=}')

    mols = compound.xyz_to_mol_all(p.xyzfilename, basis=o.basisname, ignore=True)
    atomic_numbers = moldata_read(p.xyzfilename)
    nenv = dict(zip(*get_elements_list(atomic_numbers, return_counts=True), strict=True))

    basis = Basis(o.basisname, elements=nenv.keys())
    ao_indices = [basis.index(atoms) for atoms in atomic_numbers]

    coefficients = load_coefs(len(atomic_numbers), p.coefffilebase)

    av_coefs = get_averages(nenv, basis, coefficients, ao_indices)

    for imol, (coef, mol, ao_index) in tqdm([*enumerate(zip(coefficients, mols, ao_indices, strict=True))]):

        coef = reorder.reorder_ao(mol, coef, dest='gpr', src=o.coeff_order)
        np.save(f'{p.goodcoeffilebase}{imol}.npy', coef)

        coef = remove_averages(ao_index, coef, av_coefs)

        if o.copy_metric:
            over = np.load(f'{p.overfilebase}{imol}.npy')
            over = reorder.reorder_ao(mol, over, dest='gpr', src=o.overlap_order)
            metatensor.save(f'{p.goodoverfilebase}{imol}.mts', equio.array_to_tensormap(mol, over, src='gpr'))
        else:
            over = equio.tensormap_to_array(mol, metatensor.load(f'{p.goodoverfilebase}{imol}.mts'), dest='gpr', fast=True)

        proj = over @ coef
        metatensor.save(f'{p.baselinedwbase}{imol}.mts', equio.array_to_tensormap(mol, proj, src='gpr'))
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


def get_averages(nenv, basis, coefficients, ao_indices):
    av_coefs = {q: np.zeros(basis.nmax[(q, 0)]) for q in nenv}

    for coef, ao_index in zip(coefficients, ao_indices, strict=True):
        for iat, q in enumerate(ao_index.atoms):
            av_coefs[q] += coef[ao_index.find(iat=iat, l=0)]

    for q in av_coefs:
        av_coefs[q] /= nenv[q]
    return av_coefs


if __name__=='__main__':
    main()
