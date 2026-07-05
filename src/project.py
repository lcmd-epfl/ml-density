#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import ase.io
import metatensor
from qstack import reorder
from qstack.io import metatensor as equio
from libs.config import read_config
from libs.functions import get_elements, Basis, make_dummy_mol
from libs.tmap import averages2tmap


def main():
    o, p = read_config(sys.argv)
    print(f'{o.process_metric=}')

    mol_names, atomic_numbers = prepare_molecules(p)

    nenv = dict(zip(*get_elements(atomic_numbers, return_counts=True), strict=True))

    basis = Basis(o.basisname, elements=nenv.keys())
    ao_indices = [basis.index(atoms) for atoms in atomic_numbers]

    coefficients = load_coefs(mol_names, p.input_coeffs)

    av_coefs = get_averages(nenv, basis, coefficients, ao_indices)

    for imol, (mol_name, coef, atoms, ao_index) in tqdm([*enumerate(zip(mol_names, coefficients, atomic_numbers, ao_indices, strict=True))]):

        mol = make_dummy_mol(atoms, basis=o.basisname, ignore=True)

        coef = reorder.reorder_ao(mol, coef, dest='gpr', src=o.coeff_order)
        np.save(p.clean_coefficients.format(imol), coef)
        coef = remove_averages(ao_index, coef, av_coefs)

        if o.process_metric:
            over = np.load(p.input_metrics.format(mol_name=mol_name))
            over = reorder.reorder_ao(mol, over, dest='gpr', src=o.overlap_order)
            metatensor.save(p.metric_matrix.format(imol), equio.array_to_tensormap(mol, over, src='gpr'))
        else:
            over = equio.tensormap_to_array(mol, metatensor.load(p.metric_matrix.format(imol)), dest='gpr', fast=True)

        proj = over @ coef
        metatensor.save(p.projection.format(imol), equio.array_to_tensormap(mol, proj, src='gpr'))
    metatensor.save(p.spherical_averages, averages2tmap(av_coefs))


def prepare_molecules(p):
    df = pd.read_csv(p.dataset)
    mol_names = df['id'].to_list()
    asemols = [ase.io.read(p.xyz.format(mol_name=mol_name)) for mol_name in mol_names]
    ase.io.write(p.xyzfilename, asemols)
    atomic_numbers = [asemol.numbers for asemol in asemols]
    return mol_names, atomic_numbers


def load_coefs(mol_names, fname_template):
    return [np.load(fname_template.format(mol_name=mol_name)) if (fname := fname_template.format(mol_name=mol_name)).endswith('.npy') else np.loadtxt(fname) for mol_name in mol_names]


def get_averages(nenv, basis, coefficients, ao_indices):
    av_coefs = {q: np.zeros(basis.nmax[q][0]) for q in nenv}

    for coef, ao_index in zip(coefficients, ao_indices, strict=True):
        for iat, q in enumerate(ao_index.atoms):
            av_coefs[q] += coef[ao_index.find(iat=iat, l=0)]

    for q in av_coefs:
        av_coefs[q] /= nenv[q]
    return av_coefs


def remove_averages(ao_index, coef, av_coefs):
    coef_new = np.copy(coef)
    for iat, q in enumerate(ao_index.atoms):
        coef_new[ao_index.find(iat=iat, l=0)] -= av_coefs[q]
    return coef_new


if __name__=='__main__':
    main()
