#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read,get_elements_list,get_atomicindx,basis_info,get_kernel_sizes,nao_for_mol,get_training_sets
import os
import ctypes
import ctypes_def

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    f  = conf.get_option('trainfrac', np.array([1.0]), conf.floats)
    m  = conf.get_option('m'        , 100,             int  )
    return [f,m]

[fracs,M] = set_variable_values()

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

# reference environments
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
fracs.sort()     #####
frac = fracs[-1] #####
nfrac,ntrains,train_configs = get_training_sets(trainfilename, fracs)
ntrain = ntrains[-1]
natoms_train = natoms[train_configs]
atomicindx_training = atomicindx[train_configs]
atom_counting_training = atom_counting[train_configs]
atomic_elements = np.zeros((ntrain,natmax),int)
for itrain,iconf in enumerate(train_configs):
  atomic_elements[itrain,0:natoms_train[itrain]] = element_indices[iconf]


# sparse overlap and projection indices
total_sizes = np.array([ nao_for_mol(atomic_numbers[imol], lmax, nmax) for imol in train_configs ])
# sparse kernel indices
kernel_sizes = get_kernel_sizes(train_configs, ref_elements, el_dict, M, lmax, atom_counting_training)

################################################################################

get_matrices = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/clibs/get_matrices.so")

argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_3d_int,
  ctypes_def.array_2d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_2d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_2d_int,
  ctypes.c_char_p,
  ctypes.c_char_p,
  ctypes.POINTER(ctypes.c_char_p)
  ]


bmatfiles= [f'{bmatfilebase}_M{M}_trainfrac{fracs[i]}.dat' for i in range(nfrac)]

from libs.get_matrices_B import get_b

ret = get_b(
    el_dict,
    totsize ,
    nel     ,
    llmax   ,
    nnmax   ,
    M       ,
    ntrain  ,
    natmax  ,
    nfrac,
    ntrains.astype(np.uint32)               ,
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
    goodoverfilebase,
    kernelconfbase,
    bmatfiles)
