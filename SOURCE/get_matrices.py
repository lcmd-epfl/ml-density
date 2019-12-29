#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import *

import os
import sys
import ctypes
import numpy.ctypeslib as npct

conf = Config()

def set_variable_values():
    f  = conf.get_option('trainfrac'   ,  1.0,  float)
    m  = conf.get_option('m'           ,  100,  int  )
    return [f,m]

[frac,M] = set_variable_values()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
trainfilename    = conf.paths['trainingselfile']
specselfilebase  = conf.paths['spec_sel_base']
kernelconfbase   = conf.paths['kernel_conf_base']
baselinedwbase   = conf.paths['baselined_w_base']
goodoverfilebase = conf.paths['goodover_base']
avecfilebase     = conf.paths['avec_base']
bmatfilebase     = conf.paths['bmat_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

# atomic species arrays
species = get_species_list(atomic_numbers)
nspecies = len(species)
(atom_counting, spec_list_per_conf) = get_spec_list_per_conf(species, ndata, natoms, atomic_numbers)

# atomic indices sorted by number
atomicindx = get_atomicindx(ndata,nspecies,natmax,atom_counting,spec_list_per_conf)

#====================================== reference environments
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)

# basis set size
llmax = max(lmax.values())
nnmax = max(nmax.values())
[bsize, almax, anmax] = basis_info(spe_dict, lmax, nmax);

# problem dimensionality
totsize = sum(bsize[fps_species])

# training set selection
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]

# training set arrays
train_configs = np.array(trainrange,int)
atomicindx_training = atomicindx[trainrange,:,:]

atom_counting_training = atom_counting[trainrange]
atomic_species = np.zeros((ntrain,natmax),int)
for itrain in range(ntrain):
    for iat in range(natoms_train[itrain]):
        atomic_species[itrain,iat] = spec_list_per_conf[trainrange[itrain]][iat]

# sparse overlap and projection indexes
total_sizes = np.zeros(ntrain,int)
itrain = 0
for iconf in trainrange:
    atoms = atomic_numbers[iconf]
    for iat in range(natoms[iconf]):
        for l in range(lmax[atoms[iat]]+1):
            total_sizes[itrain] += (2*l+1) * nmax[(atoms[iat],l)]
    itrain += 1

# sparse kernel indexes
kernel_sizes = get_kernel_sizes(trainrange, fps_species, spe_dict, M, lmax, atom_counting_training)

################################################################################

array_1d_int = npct.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')

get_matrices = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/get_matrices.so")
get_matrices.get_matrices.restype = ctypes.c_int
get_matrices.get_matrices.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  ctypes.c_char_p,
  ctypes.c_char_p,
  ctypes.c_char_p,
  ctypes.c_char_p,
  ctypes.c_char_p ]

ret = get_matrices.get_matrices(
    totsize ,
    nspecies,
    llmax   ,
    nnmax   ,
    M       ,
    ntrain  ,
    natmax  ,
    atomicindx_training.flatten().astype(np.uint32)   ,
    atom_counting_training.flatten().astype(np.uint32),
    train_configs.astype(np.uint32)                   ,
    natoms_train.astype(np.uint32)                    ,
    total_sizes.astype(np.uint32)                     ,
    kernel_sizes.astype(np.uint32)                    ,
    atomic_species.flatten().astype(np.uint32)        ,
    fps_species.astype(np.uint32)                     ,
    almax.astype(np.uint32)                           ,
    anmax.flatten().astype(np.uint32)                 ,
    baselinedwbase.encode('ascii'),
    goodoverfilebase.encode('ascii'),
    kernelconfbase.encode('ascii'),
    (avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt").encode('ascii'),
    (bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".dat").encode('ascii'))

