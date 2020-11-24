#!/usr/bin/env python3

import numpy as np
from config import Config
from ase.data import chemical_symbols
from basis import basis_read_full
from functions import moldata_read,averages_read,number_of_electrons_ao,correct_number_of_electrons,get_test_set

conf = Config()

def set_variable_values():
    f   = conf.get_option('trainfrac'   ,  1.0,   float)
    m   = conf.get_option('m'           ,  100,   int  )
    r   = conf.get_option('regular'     ,  1e-6,  float)
    j   = conf.get_option('jitter'      ,  1e-10, float)
    q   = conf.get_option('charges'     ,  0,     int  )  # compatibility
    return [f,m,r,j,q]

[frac,M,reg,jit,use_charges] = set_variable_values()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
trainfilename    = conf.paths['trainingselfile']
predictfilebase  = conf.paths['predict_base']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']
avdir            = conf.paths['averages_dir']
if use_charges:
  chargefilename = conf.paths['chargesfile']

# number of molecules, number of atoms in each molecule, atomic numbers
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
# basis, elements dictionary, max. angular momenta, number of radial channels
(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)

ntest,test_configs = get_test_set(trainfilename, nmol)
coeffs_unraveled = np.load(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")
av_coefs = averages_read(el_dict.values(), avdir)

if use_charges:
  print('charge_file:', chargefilename, '\n')
  molcharges = np.loadtxt(chargefilename, dtype=int)
else:
  molcharges = np.zeros(nmol, dtype=int)

error_sum = 0.0
STD_bl = 0.0
STD = 0.0

for itest,imol in enumerate(test_configs):

    atoms = atomic_numbers[imol]
    N  = sum(atoms) - molcharges[imol]
    S  = np.load(goodoverfilebase+str(imol)+".npy")
    c0 = np.load(goodcoeffilebase+str(imol)+".npy")
    qvec = number_of_electrons_ao(basis, atoms)

    c_av = np.zeros_like(c0)
    c_bl = np.zeros_like(c0)
    icoeff=0
    for iat,q in enumerate(atoms):
      for l in range(lmax[q]+1):
        msize=2*l+1
        for n in range(nmax[(q,l)]):
          if l==0:
            c_av[icoeff] = av_coefs[q][n]
          c_bl[icoeff:icoeff+msize] = coeffs_unraveled[itest,iat,l,n,0:msize]
          icoeff+=msize

    #================================================
    c0_bl = c0   - c_av
    c     = c_bl + c_av
    dc    = c_bl - c0_bl

    nel_ref = qvec @ c0
    nel_pr  = qvec @ c

    error    = dc    @ S @ dc
    norm_bl  = c0_bl @ S @ c0_bl
    norm     = c0    @ S @ c0

    if use_charges:
      cn     = correct_number_of_electrons(c, S, qvec, N)
      dcn    = cn - c0
      errorn = dcn @ S @ dcn
    else:
      errorn = np.nan

    error_sum += error
    STD_bl    += norm_bl
    STD       += norm
    strg = "mol # %*i (%*i):  %8.3f %%  %.2e %%    ( %.2e )   %8.4f / %8.4f ( %3d )     (corr N: %8.3f %%)"%(
        len(str(ntest)),
        itest,
        len(str(nmol)),
        imol,
        (error/norm_bl)*100.0,
        (error/norm)*100.0,
        error,
        nel_pr,
        nel_ref,
        N,
        (errorn/norm_bl)*100.0
        )
    print(strg)

print()
print("%% RMSE = %.2e %%  %.2e %%    ( %.2e )" % (
      (error_sum/STD_bl)*100.0,
      (error_sum/STD)*100.0,
      error_sum/ntest
))

