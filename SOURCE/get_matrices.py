#!/usr/bin/python

import numpy as np
import time
import ase.io
import get_matrices
from config import Config
from basis import basis_read_num


def basis_info(spe_dict, lmax, nmax):
    nspecies = len(spe_dict)
    llmax = max(lmax.values())
    bsize = np.zeros(nspecies,int)
    almax = np.zeros(nspecies,int)
    anmax = np.zeros((nspecies,llmax+1),int)
    for ispe in xrange(nspecies):
        spe = spe_dict[ispe]
        almax[ispe] = lmax[spe]+1
        for l in xrange(lmax[spe]+1):
            anmax[ispe,l] = nmax[(spe,l)]
            bsize[ispe] += nmax[(spe,l)]*(2*l+1)
    return [bsize, almax, anmax]

def get_kernel_sizes( trainrange, fps_species, spe_dict, M, lmax, atom_counting_training ):
    kernel_sizes = np.zeros(len(trainrange),int)
    itrain = 0
    for iconf in trainrange:
        for iref in xrange(M):
            ispe = fps_species[iref]
            spe = spe_dict[ispe]
            temp = 0
            for l in xrange(lmax[spe]+1):
                msize = 2*l+1
                temp += msize*msize
            kernel_sizes[itrain] += temp * atom_counting_training[itrain,ispe]
        itrain += 1
    return kernel_sizes

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


# system definition
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)

# system parameters
atomic_numbers = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_numbers.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_numbers[i]))
natmax = max(natoms)

# atomic species arrays
species = np.sort(list(set(np.array([item for sublist in atomic_numbers for item in sublist]))))
nspecies = len(species)

spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_numbers[iconf][iat] == species[ispe]:
               atom_counting[iconf,ispe] += 1
               spec_list_per_conf[iconf].append(ispe)
nenv = sum(natoms)

# atomic indexes sorted by number
atomicindx = np.zeros((natmax,nspecies,ndata),int)
for iconf in xrange(ndata):
    for ispe in xrange(nspecies):
        indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
        for icount in xrange(atom_counting[iconf,ispe]):
            atomicindx[icount,ispe,iconf] = indexes[icount]


#====================================== reference environments
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read_num(basisfilename)

if len(spe_dict) != nspecies:
    print "different number of elements in the molecules and in the basis"
    exit(0)

# basis set size
llmax = max(lmax.values())
nnmax = max(nmax.values())
[bsize, almax, anmax] = basis_info(spe_dict, lmax, nmax);

# problem dimensionality
totsize = 0
for iref in xrange(0,M):
    totsize += bsize[fps_species[iref]]


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
np.save(avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".npy", Avec)
np.save(bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".npy", Bmat)

