#!/usr/bin/env python3

import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,get_elements_list,get_atomicindx,basis_info,get_kernel_sizes

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

# elements array and atomic indices sorted by elements
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

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
[bsize, alnum, annum] = basis_info(el_dict, lmax, nmax);

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
        atomic_elements[itrain,iat] = element_indices[trainrange[itrain]][iat]

# sparse overlap and projection indices
total_sizes = np.zeros(ntrain,int)
itrain = 0
for imol in trainrange:
    atoms = atomic_numbers[imol]
    for iat in range(natoms[imol]):
        for l in range(lmax[atoms[iat]]+1):
            total_sizes[itrain] += (2*l+1) * nmax[(atoms[iat],l)]
    itrain += 1

# sparse kernel indices
kernel_sizes = get_kernel_sizes(trainrange, ref_elements, el_dict, M, lmax, atom_counting_training)

################################################################################

array_1d_int = npct.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.uint32, ndim=2, flags='CONTIGUOUS')
array_3d_int = npct.ndpointer(dtype=np.uint32, ndim=3, flags='CONTIGUOUS')
get_matrices = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/get_matrices.so")

argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  array_3d_int,
  array_2d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_2d_int,
  array_1d_int,
  array_1d_int,
  array_2d_int,
  ctypes.c_char_p,
  ctypes.c_char_p,
  ctypes.c_char_p ]


if len(sys.argv)>1 and sys.argv[1][0].lower()=='b':

  get_matrices.get_b.restype = ctypes.c_int
  get_matrices.get_b.argtypes = argtypes

  ret = get_matrices.get_b(
      totsize ,
      nel     ,
      llmax   ,
      nnmax   ,
      M       ,
      ntrain  ,
      natmax  ,
      atomicindx_training.astype(np.uint32)   ,
      atom_counting_training.astype(np.uint32),
      train_configs.astype(np.uint32)         ,
      natoms_train.astype(np.uint32)          ,
      total_sizes.astype(np.uint32)           ,
      kernel_sizes.astype(np.uint32)          ,
      atomic_elements.astype(np.uint32)       ,
      ref_elements.astype(np.uint32)          ,
      alnum.astype(np.uint32)                 ,
      annum.astype(np.uint32)                 ,
      goodoverfilebase.encode('ascii'),
      kernelconfbase.encode('ascii'),
      (bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".dat").encode('ascii'))

else:
  get_matrices.get_a.restype = ctypes.c_int
  get_matrices.get_a.argtypes = argtypes

  ret = get_matrices.get_a(
      totsize ,
      nel     ,
      llmax   ,
      nnmax   ,
      M       ,
      ntrain  ,
      natmax  ,
      atomicindx_training.astype(np.uint32)   ,
      atom_counting_training.astype(np.uint32),
      train_configs.astype(np.uint32)         ,
      natoms_train.astype(np.uint32)          ,
      total_sizes.astype(np.uint32)           ,
      kernel_sizes.astype(np.uint32)          ,
      atomic_elements.astype(np.uint32)       ,
      ref_elements.astype(np.uint32)          ,
      alnum.astype(np.uint32)                 ,
      annum.astype(np.uint32)                 ,
      baselinedwbase.encode('ascii'),
      kernelconfbase.encode('ascii'),
      (avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt").encode('ascii'))

