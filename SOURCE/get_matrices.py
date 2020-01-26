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
elselfilebase    = conf.paths['spec_sel_base']
kernelconfbase   = conf.paths['kernel_conf_base']
baselinedwbase   = conf.paths['baselined_w_base']
goodoverfilebase = conf.paths['goodover_base']
avecfilebase     = conf.paths['avec_base']
bmatfilebase     = conf.paths['bmat_base']


(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

# atomic elements arrays
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atom_counting, el_list_per_conf) = get_el_list_per_conf(elements, nmol, natoms, atomic_numbers)

# atomic indices sorted by number
atomicindx = get_atomicindx(nmol,nel,natmax,atom_counting,el_list_per_conf)

#====================================== reference environments
ref_elements = np.loadtxt(elselfilebase+str(M)+".txt",int)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)

# basis set size
llmax = max(lmax.values())
nnmax = max(nmax.values())
[bsize, almax, anmax] = basis_info(el_dict, lmax, nmax);

# problem dimensionality
totsize = sum(bsize[ref_elements])

# training set selection
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]

# training set arrays
train_configs = np.array(trainrange,int)
atomicindx_training = atomicindx[trainrange,:,:]

atom_counting_training = atom_counting[trainrange]
atomic_elements = np.zeros((ntrain,natmax),int)
for itrain in range(ntrain):
    for iat in range(natoms_train[itrain]):
        atomic_elements[itrain,iat] = el_list_per_conf[trainrange[itrain]][iat]

# sparse overlap and projection indexes
total_sizes = np.zeros(ntrain,int)
itrain = 0
for imol in trainrange:
    atoms = atomic_numbers[imol]
    for iat in range(natoms[imol]):
        for l in range(lmax[atoms[iat]]+1):
            total_sizes[itrain] += (2*l+1) * nmax[(atoms[iat],l)]
    itrain += 1

# sparse kernel indexes
kernel_sizes = get_kernel_sizes(trainrange, ref_elements, el_dict, M, lmax, atom_counting_training)

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
    nel     ,
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
    atomic_elements.flatten().astype(np.uint32)        ,
    ref_elements.astype(np.uint32)                     ,
    almax.astype(np.uint32)                           ,
    anmax.flatten().astype(np.uint32)                 ,
    baselinedwbase.encode('ascii'),
    goodoverfilebase.encode('ascii'),
    kernelconfbase.encode('ascii'),
    (avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt").encode('ascii'),
    (bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".dat").encode('ascii'))

