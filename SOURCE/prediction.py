#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read,get_elements_list,get_atomicindx,get_test_set,get_training_set
from run_prediction import run_prediction

path = get_config_path(sys.argv)
conf = Config(config_path=path)

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
elselfilebase   = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
weightsfilebase = conf.paths['weights_base']
predictfilebase = conf.paths['predict_base']

training = 'training' in sys.argv[1:]

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

# elements array and atomic indices sorted by elements
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

if not training:
  ntest,test_configs = get_test_set(trainfilename, nmol)
  predictfile = predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy"
else:
  ntest,test_configs = get_training_set(trainfilename, frac)
  predictfile = predictfilebase + "_training_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy"
weightsfile = weightsfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy"

natoms_test = natoms[test_configs]
atomicindx_test = atomicindx[test_configs]
atom_counting_test = atom_counting[test_configs]

print("Number of testing molecules =", ntest)

run_prediction(ntest, natmax, natoms_test,
    atom_counting_test, atomicindx_test, test_configs,
    M, elements,
    kernelconfbase,
    basisfilename,
    elselfilebase+str(M)+".txt",
    weightsfile,
    predictfile)

