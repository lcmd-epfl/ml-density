#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import metatensor
from qstack.io.metatensor import split, tensormap_to_array
from qstack.fields.moments import r2_c as rho_moments
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, get_training_set, Basis, make_dummy_mol
from libs.tmap import sph2vector


def correct_number_of_electrons(c, S, q, N):
    S1q = np.linalg.solve(S, q)
    return c + S1q * (N - c@q)/(q@S1q)


def get_number_of_electrons(use_charges, atomic_numbers, df):
    if use_charges in [None, 'charge']:
        nuc_charges = np.array([sum(atoms) for atoms in atomic_numbers])
        if use_charges is None:
            return nuc_charges
    inp = df[use_charges].to_numpy()
    if use_charges=='N':
        return inp
    elif use_charges=='charge':
        return nuc_charges - inp


def main():
    args, o, p = read_config(sys.argv, return_args=['training'])

    df = pd.read_csv(p.dataset)
    averages = metatensor.load(p.spherical_averages)
    atomic_numbers = moldata_read(p.xyzfilename)
    basis = Basis(o.basisname, elements=averages.keys.column('center_type'))
    N_all = get_number_of_electrons(o.use_charges, atomic_numbers, df)

    for frac in o.fracs:

        print(f'fraction = {frac}')
        if args.training:
            ntest, test_configs = get_training_set(p.train_test_sets, frac)
            predictfile = p.predictions.format(subset='training', train_frac=frac)
        else:
            ntest, test_configs = get_test_set(p.train_test_sets)
            predictfile = p.predictions.format(subset='test', train_frac=frac)
        predictions = split(metatensor.load(predictfile))

        total_error_N      = 0.0
        total_error_abs    = 0.0
        total_error_rel    = 0.0
        total_error_rel_bl = 0.0

        for itest, imol in enumerate(test_configs):

            atoms = atomic_numbers[imol]
            N = N_all[imol]
            mol = make_dummy_mol(atoms=atoms, basis=o.basisname, charge=sum(atoms)-N, spin=N%2)
            qvec = rho_moments(mol, rho=None, moments=(0,), per_atom=False)[0]

            S    = tensormap_to_array(mol, metatensor.load(p.metric_matrix.format(imol)), dest='gpr', fast=True)
            c0   = np.load(p.clean_coefficients.format(imol))
            c_bl = tensormap_to_array(mol, predictions[itest], dest='gpr', fast=True)
            c_av = sph2vector(atoms, basis, averages)

            c0_bl = c0 - c_av
            nel0  = qvec @ c0

            c   = c_bl + c_av
            dc  = c0 - c
            nel = qvec @ c

            error    = dc    @ S @ dc
            norm     = c0    @ S @ c0
            norm_bl  = c0_bl @ S @ c0_bl
            error_rel_bl = error/norm_bl * 100.0
            error_rel    = error/norm * 100.0
            total_error_abs    += error
            total_error_rel    += error_rel
            total_error_rel_bl += error_rel_bl

            if o.use_charges:
                cn     = correct_number_of_electrons(c, S, qvec, N)
                dcn    = cn - c0
                errorn = dcn @ S @ dcn
                errorn_rel_bl = errorn / norm_bl * 100.0
                total_error_N += abs(nel - N)
            else:
                errorn_rel_bl = np.nan

            s1 = f'mol # {itest:{len(str(ntest))}} ({imol:{len(str(len(atomic_numbers)))}}):  '
            s2 = f'{error_rel_bl:8.3f} %  {error_rel:.2e} %    ( {error:.2e} )   {nel:8.4f} / {nel0:8.4f} ( {N:3d} )     (corr N: {errorn_rel_bl:8.3f} %)    {p.xyz.format(mol_name=df['id'][imol])}'
            print(s1+s2)

        print(f'\nfrac={frac}\tMAE = {total_error_rel_bl/ntest:.2e} %  {total_error_rel/ntest:.2e} %    ( {total_error_abs/ntest:.2e} )', end='')

        if o.use_charges:
            print(f'  ΔN: {total_error_N/ntest:.2e}')
        else:
            print()


if __name__=='__main__':
    main()
