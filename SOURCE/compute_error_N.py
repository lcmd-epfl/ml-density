#!/usr/bin/env python3

import numpy as np
from config import Config
from ase.data import chemical_symbols
from basis import basis_read_full
from functions import moldata_read,averages_read,number_of_electrons_ao,correct_number_of_electrons

conf = Config()

def set_variable_values():
    f   = conf.get_option('trainfrac'   ,  1.0,   float)
    m   = conf.get_option('m'           ,  100,   int  )
    r   = conf.get_option('regular'     ,  1e-6,  float)
    j   = conf.get_option('jitter'      ,  1e-10, float)
    return [f,m,r,j]

[frac,M,reg,jit] = set_variable_values()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
trainfilename    = conf.paths['trainingselfile']
predictfilebase  = conf.paths['predict_base']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']
avdir            = conf.paths['averages_dir']
chargefilename   = conf.paths['chargesfile']

# number of molecules, number of atoms in each molecule, atomic numbers
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)

# basis, elements dictionary, max. angular momenta, number of radial channels
(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)

# load predicted coefficients for test structures
trainrangetot = np.loadtxt(trainfilename,int)
testrange = np.setdiff1d(range(nmol),trainrangetot)

coeffs_unraveled = np.load(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

av_coefs = averages_read(el_dict.values(), avdir)
charges  = np.loadtxt(chargefilename, dtype=int)

error_sum = 0.0
STD_bl = 0.0
STD = 0.0

for itest,imol in enumerate(testrange):

    atoms = atomic_numbers[imol]
    N  = sum(atoms) - charges[imol]
    S  = np.load(goodoverfilebase+str(imol)+".npy")
    c0 = np.load(goodcoeffilebase+str(imol)+".npy")
    q  = number_of_electrons_ao(basis, atoms)

    c_av = np.zeros_like(c0)
    c_bl = np.zeros_like(c0)
    icoeff = 0
    for iat in range(natoms[imol]):
        for l in range(lmax[atoms[iat]]+1):
            for n in range(nmax[(atoms[iat],l)]):
                for im in range(2*l+1):
                    if l==0:
                        c_av[icoeff] = av_coefs[atoms[iat]][n]
                    c_bl[icoeff] = coeffs_unraveled[itest,iat,l,n,im]
                    icoeff +=1

    #================================================
    c0_bl = c0   - c_av
    c     = c_bl + c_av
    dc    = c_bl - c0_bl

    nel_ref = q @ c0
    nel_pr  = q @ c

    error    = dc    @ S @ dc
    norm_bl  = c0_bl @ S @ c0_bl
    norm     = c0    @ S @ c0

    cn     = correct_number_of_electrons(c, S, q, N)
    dcn    = cn - c0
    errorn = dcn @ S @ dcn

    error_sum += error
    STD_bl    += norm_bl
    STD       += norm
    strg = "mol # %*i (%*i):  %8.3f %%  %.2e %%    ( %.2e )   %8.4f / %8.4f ( %3d )     (N %8.3f %%)"%(
        len(str(len(testrange))),
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
      error_sum/len(testrange)
))

