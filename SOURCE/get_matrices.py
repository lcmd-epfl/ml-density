#!/usr/bin/python

import numpy as np
import time
import get_matrices
from config import Config
from basis import basis_read
from functions import *


conf = Config()

def set_variable_values():
    f  = conf.get_option('trainfrac'   ,  1.0,  float)
    m  = conf.get_option('m'           ,  100,  int  )
    return [f,m]

[frac,M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
trainfilename   = conf.paths['trainingselfile']
specselfilebase = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
baselinedwbase  = conf.paths['baselined_w_base']
overdatbase     = conf.paths['over_dat_base']
avecfilebase    = conf.paths['avec_base']
bmatfilebase    = conf.paths['bmat_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

# atomic species arrays
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

# problem dimensionality
totsize = sum(bsize[fps_species])
print "problem dimensionality =", totsize

# training set selection
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
print "Number of training molecules = ", ntrain

# training set arrays
train_configs = np.array(trainrange,int)
atomicindx_training = atomicindx[:,:,trainrange]
atom_counting_training = atom_counting[trainrange]
atomic_species = np.zeros((ntrain,natmax),int)
for itrain in xrange(ntrain):
    for iat in xrange(natoms_train[itrain]):
        atomic_species[itrain,iat] = spec_list_per_conf[trainrange[itrain]][iat]

# sparse overlap and projection indexes
total_sizes = np.zeros(ntrain,int)
itrain = 0
for iconf in trainrange:
    atoms = atomic_numbers[iconf]
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            total_sizes[itrain] += (2*l+1) * nmax[(atoms[iat],l)]
    itrain += 1

# sparse kernel indexes
kernel_sizes = get_kernel_sizes(trainrange, fps_species, spe_dict, M, lmax, atom_counting_training)

# compute regression arrays
start = time.time()
Avec,Bmat = get_matrices.getab(baselinedwbase, overdatbase, kernelconfbase,
                               train_configs,atomic_species,llmax,nnmax,nspecies,ntrain,M,natmax,natoms_train,totsize,
                               atomicindx_training,atom_counting_training,fps_species,almax,anmax,total_sizes,kernel_sizes)
print "A-vector and B-matrix computed in", time.time()-start, "seconds"

# save regression arrays
np.savetxt(avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt", Avec)
np.savetxt(bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt", Bmat)

