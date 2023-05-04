#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import read_config
from basis import basis_read_full
from functions import moldata_read, number_of_electrons_ao, correct_number_of_electrons, get_test_set, get_training_set


def main():
    o, p = read_config(sys.argv)
    training = 'training' in sys.argv[1:]

    # number of molecules, number of atoms in each molecule, atomic numbers
    (nmol, natoms, atomic_numbers) = moldata_read(p.xyzfilename)
    # basis, elements dictionary, max. angular momenta, number of radial channels
    (basis, el_dict, lmax, nmax) = basis_read_full(p.basisfilename)

    averages = equistore.load(p.avfile)

    if o.use_charges:
        print(f'charge_file: {p.chargefilename} mode: {o.use_charges}\n')
        molcharges = np.loadtxt(p.chargefilename, dtype=int)

    for frac in o.fracs:

      print(f'fraction = {frac}')

      if not training:
          ntest, test_configs = get_test_set(p.trainfilename, nmol)
          predictfile = f'{p.predictfilebase}_test_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
      else:
          ntest, test_configs = get_training_set(p.trainfilename, frac)
          predictfile = f'{p.predictfilebase}_training_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
      coeffs_unraveled = np.load(predictfile)

      dn_av = 0.0
      total_error_N      = 0.0
      total_error_abs    = 0.0
      total_error_rel    = 0.0
      total_error_rel_bl = 0.0

      for itest,imol in enumerate(test_configs):

          atoms = atomic_numbers[imol]
          if o.use_charges==0:
              N  = sum(atoms)
          elif o.use_charges==1:
              N  = sum(atoms) - molcharges[imol]
          elif o.use_charges==2:
              N  = molcharges[imol]

          S  = np.load(f'{p.goodoverfilebase}{imol}.npy')
          c0 = np.load(f'{p.goodcoeffilebase}{imol}.npy')
          qvec = number_of_electrons_ao(basis, atoms)

          c_av = np.zeros_like(c0)
          c_bl = np.zeros_like(c0)
          icoeff=0
          for iat, q in enumerate(atoms):
              c_av[icoeff:icoeff+nmax[(q,0)]] = averages.block(spherical_harmonics_l=0, species_center=q).values
              for l in range(lmax[q]+1):
                  msize = 2*l+1
                  nsize = nmax[(q,l)]
                  c_bl[icoeff:icoeff+nsize*msize] = coeffs_unraveled[itest, iat, l, 0:nsize, 0:msize].flatten()
                  icoeff+=nsize*msize

          #================================================
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

      print()

      print(f'% MAE = {total_error_rel_bl/ntest:.2e} %  {total_error_rel/ntest:.2e} %    ( {total_error_abs/ntest:.2e} )')

      if o.use_charges:
          print(f'{total_error_N/ntest}')
      print()


if __name__=='__main__':
    main()
