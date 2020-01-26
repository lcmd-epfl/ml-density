#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read_full
from ase.data import chemical_symbols
from functions import moldata_read,number_of_electrons

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

# number of molecules, number of atoms in each molecule, atomic numbers
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)

# elements dictionary, max. angular momenta, number of radial channels
(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)

# load predicted coefficients for test structures
trainrangetot = np.loadtxt(trainfilename,int)
testrange = np.setdiff1d(range(nmol),trainrangetot)

coeffs_unraveled = np.load(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

av_coefs = {}
for q in el_dict.values():
    av_coefs[q] = np.load(avdir+chemical_symbols[q]+".npy")

itest = 0
error_sum = 0.0
STD_bl = 0.0
STD = 0.0
for imol in testrange:
    atoms = atomic_numbers[imol]
    #================================================
    overl      = np.load(goodoverfilebase+str(imol)+".npy")
    coeffs_ref = np.load(goodcoeffilebase+str(imol)+".npy")
    size_coeffs = coeffs_ref.shape


    averages  = np.zeros(size_coeffs,float)
    coeffs_bl = np.zeros(size_coeffs,float)
    icoeff = 0
    for iat in range(natoms[imol]):
        for l in range(lmax[atoms[iat]]+1):
            for n in range(nmax[(atoms[iat],l)]):
                for im in range(2*l+1):
                    if l==0:
                        averages[icoeff] = av_coefs[atoms[iat]][n]
                    coeffs_bl[icoeff] = coeffs_unraveled[itest,iat,l,n,im]
                    icoeff +=1

    coeffs_ref_bl = coeffs_ref - averages
    delta_coeffs  = coeffs_bl  - coeffs_ref_bl

    #================================================

    nel_ref = number_of_electrons(basis, atoms, coeffs_ref)
    nel_pr  = number_of_electrons(basis, atoms, coeffs_bl+averages)

    error    = np.dot(delta_coeffs , np.dot(overl, delta_coeffs ))
    norm_bl  = np.dot(coeffs_ref_bl, np.dot(overl, coeffs_ref_bl))
    norm     = np.dot(coeffs_ref   , np.dot(overl, coeffs_ref   ))
    error_sum += error
    STD_bl    += norm_bl
    STD       += norm
    strg = "mol # %*i (%*i):  %8.3f %%  %.2e %%    ( %.2e )   %8.4f / %8.4f ( %3d )"%(
        len(str(len(testrange))),
        itest,
        len(str(nmol)),
        imol,
        (error/norm_bl)*100.0,
        (error/norm)*100.0,
        error,
        nel_pr,
        nel_ref,
        sum(atoms)
        )
    print(strg)
    itest += 1

print()
print("%% RMSE = %.2e %%  %.2e %%    ( %.2e )" % (
      (error_sum/STD_bl)*100.0,
      (error_sum/STD)*100.0,
      error_sum/len(testrange)
))

