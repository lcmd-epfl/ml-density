#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read,get_elements_list,get_atomicindx,basis_info,get_kernel_sizes,nao_for_mol,get_training_set
import os
import ctypes
import ctypes_def

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    m  = conf.get_option('m'           ,  100,  int  )
    return [m]

[M] = set_variable_values()

xyzfilename    = conf.paths['xyzfile']
basisfilename  = conf.paths['basisfile']
trainfilename  = conf.paths['trainingselfile']
elselfilebase  = conf.paths['spec_sel_base']
kernelconfbase = conf.paths['kernel_conf_base']
qfilebase      = conf.paths['charges_base']
Kqfilebase     = conf.paths['kernel_charges_base']

elselfile   = elselfilebase+str(M)+".txt"
Kqfile      = Kqfilebase+"_M"+str(M)+".dat"

#======================================
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

# elements array and atomic indices sorted by elements
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

# reference environments
ref_elements = np.loadtxt(elselfile, int)

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
ntrain,train_configs = get_training_set(trainfilename, sort=False)
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

get_matrices = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/get_matrices.so")

argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
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
  ctypes.c_char_p ]

get_matrices.get_a_for_mol.restype = ctypes.c_int
get_matrices.get_a_for_mol.argtypes = argtypes

ret = get_matrices.get_a_for_mol(
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
    qfilebase.encode('ascii'),
    kernelconfbase.encode('ascii'),
    Kqfile.encode('ascii'))

