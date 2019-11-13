#!/usr/bin/python

import numpy as np
import prediction
from config import Config
from basis import basis_read
from functions import *

conf = Config()

def set_variable_values():
    f   = conf.get_option('trainfrac'   ,  1.0,   float)
    m   = conf.get_option('m'           ,  100,   int  )
    r   = conf.get_option('regular'     ,  1e-6,  float)
    j   = conf.get_option('jitter'      ,  1e-10, float)
    return [f,m,r,j]

[frac,M,reg,jit] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
trainfilename   = conf.paths['trainingselfile']
specselfilebase = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
weightsfilebase = conf.paths['weights_base']
predictfilebase = conf.paths['predict_base']

(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)

(atom_counting, spec_list_per_conf) = get_spec_list_per_conf(species, ndata, natoms, atomic_numbers)

# atomic indices sorted by number
atomicindx = get_atomicindx(ndata,nspecies,natmax,atom_counting,spec_list_per_conf)
atomicindx = atomicindx.T

#====================================== reference environments
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != spe_dict.values():
    print "different elements in the molecules and in the basis"
    exit(1)

# basis set size
llmax = max(lmax.values())
nnmax = max(nmax.values())
[bsize, almax, anmax] = basis_info(spe_dict, lmax, nmax);

# dataset partitioning
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
testrange = np.setdiff1d(range(ndata),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]
print "Number of training molecules = ", ntrain
print "Number of testing molecules = ", ntest

# define testing indexes
test_configs = np.array(testrange,int)
atomicindx_test = atomicindx[:,:,testrange]
atom_counting_test = atom_counting[testrange]
test_species = np.zeros((ntest,natmax),int)
for itest in xrange(ntest):
    for iat in xrange(natoms_test[itest]):
        test_species[itest,iat] = spec_list_per_conf[testrange[itest]][iat]

# sparse kernel sizes
kernel_sizes = get_kernel_sizes(testrange, fps_species, spe_dict, M, lmax, atom_counting_test)

# load regression weights
weights = np.load(weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

# unravel regression weights with explicit indexing
ww = np.zeros((M,llmax+1,nnmax,2*llmax+1),float)
i = 0
for ienv in xrange(M):
    ispe = fps_species[ienv]
    al = almax[ispe]
    for l in xrange(al):
        msize = 2*l+1
        anc = anmax[ispe,l]
        for n in xrange(anc):
            for im in xrange(msize):
                ww[ienv,l,n,im] = weights[i]
                i += 1

# load testing kernels and perform prediction
coeffs = prediction.prediction(kernelconfbase,
                               kernel_sizes,fps_species,atom_counting_test,atomicindx_test,nspecies,ntest,natmax,
                               llmax,nnmax,natoms_test,test_configs,test_species,almax,anmax,M,ww)

np.save(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",coeffs)

