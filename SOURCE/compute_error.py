#!/usr/bin/python

import numpy as np
import ase.io
import argparse
from config import Config
from basis import basis_read

conf = Config()

def set_variable_values():
    s   = conf.get_option('testset'     ,  1,     int  )
    f   = conf.get_option('trainfrac'   ,  1.0,   float)
    m   = conf.get_option('m'           ,  100,   int  )
    rc  = conf.get_option('cutoffradius',  4.0,   float)
    sg  = conf.get_option('sigmasoap'   ,  0.3,   float)
    r   = conf.get_option('regular'     ,  1e-6,  float)
    j   = conf.get_option('jitter'      ,  1e-10, float)
    return [s,f,m,rc,sg,r,j]

[nset,frac,M,rc,sigma_soap,reg,jit] = set_variable_values()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
trainfilename    = conf.paths['trainingselfile']
refsselfilebase  = conf.paths['refs_sel_base']
specselfilebase  = conf.paths['spec_sel_base']
predictfilebase  = conf.paths['predict_base']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']
avdir            = conf.paths['averages_dir']

bohr2ang = 0.529177249
#========================== system definition
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)
#======================= system parameters
coords = []
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    coords.append(np.asarray(xyzfile[i].get_positions(),float)/bohr2ang)
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

#====================================== reference environments
fps_indexes = np.loadtxt(refsselfilebase+str(M)+".txt",int)
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)

# load predicted coefficients for test structures
trainrangetot = np.loadtxt(trainfilename,int)
testrange = np.setdiff1d(range(ndata),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]

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
    coordinates = coords[iconf]
    atoms = atomic_symbols[iconf]
    valences = atomic_valence[iconf]
    nele = np.sum(valences)
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
    print "error =", np.sqrt(error/norm)*100, "%"
    itest+=1

print ""
print "% RMSE = ", 100*np.sqrt(error_density/STD)
