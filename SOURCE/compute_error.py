#!/usr/bin/python

import numpy as np
import ase.io
import argparse
from config import Config
from basis import basis_read

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

#========================== system definition
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)
#======================= system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = len(atomic_symbols[i])

#====================================== reference environments

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)

# load predicted coefficients for test structures
trainrangetot = np.loadtxt(trainfilename,int)
testrange = np.setdiff1d(range(ndata),trainrangetot)

coeffs = np.load(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

av_coefs = {}
for spe in spe_dict.values():
    av_coefs[spe] = np.load(avdir+str(spe)+".npy")

itest=0
error_density = 0.0
STD = 0.0
for iconf in testrange:
    print "-------------------------------"
    print "iconf = ", iconf
    atoms = atomic_symbols[iconf]
    #================================================

    overl      = np.load(goodoverfilebase+str(iconf)+".npy")
    coeffs_ref = np.load(goodcoeffilebase+str(iconf)+".npy")
    size_coeffs = coeffs_ref.shape

    averages = np.zeros(size_coeffs,float)
    icoeff = 0
    for iat in xrange(natoms[iconf]):
        for n in xrange(nmax[(atoms[iat],0)]):
            averages[icoeff] = av_coefs[atoms[iat]][n]
            icoeff +=1
        for l in xrange(1, lmax[atoms[iat]]+1):
            icoeff += (2*l+1) * nmax[(atoms[iat],l)]
    coeffs_ref  -= averages
    projs_ref    = np.dot(overl,coeffs_ref)

    delta_coeffs = np.zeros(size_coeffs,float)
    icoeff = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            for n in xrange(nmax[(atoms[iat],l)]):
                for im in xrange(2*l+1):
                    delta_coeffs[icoeff] = coeffs[itest,iat,l,n,im] - coeffs_ref[icoeff]
                    icoeff +=1
    delta_proj   = np.dot(overl,delta_coeffs)

    #================================================
    error = np.dot(delta_coeffs, delta_proj)
    error_density += error
    norm = np.dot(coeffs_ref,projs_ref)
    STD += norm
    print("error = %8.3f%%" % (np.sqrt(error/norm)*100.0))
    itest+=1

print
print("%% RMSE = %8.3f%%" % (np.sqrt(error_density/STD)*100.0))

