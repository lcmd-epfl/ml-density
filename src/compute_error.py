#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from pyscf.gto import M as make_pyscf_mol
from qstack.fields.moments import r2_c as rho_moments
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, get_training_set, Basis
from libs.tmap import split, tmap2vector, tmap2matrix, sph2vector


def correct_number_of_electrons(c, S, q, N):
    S1q = np.linalg.solve(S, q)
    return c + S1q * (N - c@q)/(q@S1q)


def main():
    o, p = read_config(sys.argv)
    training = 'training' in sys.argv[1:]

    averages = metatensor.load(p.avfile)
    atomic_numbers = moldata_read(p.xyzfilename)
    basis = Basis(o.basisname, elements=set(averages.keys.column('center_type')))
    nmol = len(atomic_numbers)

    if o.use_charges:
        print(f'charge_file: {p.chargefilename} mode: {o.use_charges}\n')
        molcharges = np.loadtxt(p.chargefilename, dtype=int)

    for frac in o.fracs:

        print(f'fraction = {frac}')
        if not training:
            ntest, test_configs = get_test_set(p.trainfilename, nmol)
            predictfile = f'{p.predictfilebase}_test_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.mts'
        else:
            ntest, test_configs = get_training_set(p.trainfilename, frac)
            predictfile = f'{p.predictfilebase}_training_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.mts'
        predictions = split(metatensor.load(predictfile))

        total_error_N      = 0.0
        total_error_abs    = 0.0
        total_error_rel    = 0.0
        total_error_rel_bl = 0.0

        for itest, imol in enumerate(test_configs):

            atoms = atomic_numbers[imol]
            if o.use_charges==0:
                N  = sum(atoms)
            elif o.use_charges==1:
                N  = sum(atoms) - molcharges[imol]
            elif o.use_charges==2:
                N  = molcharges[imol]

            S = tmap2matrix(atoms, basis, metatensor.load(f'{p.goodoverfilebase}{imol}.mts'))

            mol = make_pyscf_mol(atom=[[q, (0, 0, 0)]  for q in atoms], basis=o.basisname, charge=sum(atoms)-N, spin=N%2)
            qvec = rho_moments(mol, rho=None, moments=(0,), per_atom=False)[0]

            c0   = np.load(f'{p.goodcoeffilebase}{imol}.npy')
            c_bl = tmap2vector(atoms, basis, predictions[itest])
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

            s1 = f'mol # {itest:{len(str(ntest))}} ({imol:{len(str(nmol))}}):  '
            s2 = f'{error_rel_bl:8.3f} %  {error_rel:.2e} %    ( {error:.2e} )   {nel:8.4f} / {nel0:8.4f} ( {N:3d} )     (corr N: {errorn_rel_bl:8.3f} %)'
            print(s1+s2)

        print(f'\n{frac=}\tMAE = {total_error_rel_bl/ntest:.2e} %  {total_error_rel/ntest:.2e} %    ( {total_error_abs/ntest:.2e} )', end='')

        if o.use_charges:
            print(f'  ΔN: {total_error_N/ntest:.2e}')
        else:
            print()


if __name__=='__main__':
    main()
