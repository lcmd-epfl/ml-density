#!/usr/bin/env python3

import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,averages_read,prediction2coefficients,print_progress

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
avdir            = conf.paths['averages_dir']
outfilebase      = conf.paths['output_base']

# number of molecules, number of atoms in each molecule, atomic numbers
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)

# load predicted coefficients for test structures
trainrangetot = np.loadtxt(trainfilename,int)
testrange = np.setdiff1d(range(nmol),trainrangetot)

coeff = np.load(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

av_coefs = averages_read(el_dict.values(), avdir)

for itest,imol in enumerate(testrange):
    print_progress(itest, len(testrange))
    rho1 = prediction2coefficients(atomic_numbers[imol], lmax, nmax, coeff[itest], av_coefs, True)
    rho2 = prediction2coefficients(atomic_numbers[imol], lmax, nmax, coeff[itest], av_coefs, False)
    np.savetxt(outfilebase             +str(imol)+'.dat', rho1)
    np.savetxt(outfilebase+'noreorder_'+str(imol)+'.dat', rho2)

