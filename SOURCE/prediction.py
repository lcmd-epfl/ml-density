#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,get_elements_list,get_el_list_per_conf,get_atomicindx
from run_prediction import *

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
elselfilebase   = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
weightsfilebase = conf.paths['weights_base']
predictfilebase = conf.paths['predict_base']

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

elements = get_elements_list(atomic_numbers)
(atom_counting, el_list_per_conf) = get_el_list_per_conf(elements, nmol, natoms, atomic_numbers)

# atomic indices sorted by number
atomicindx = get_atomicindx(nmol,len(elements),natmax,atom_counting,el_list_per_conf)

# dataset partitioning
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
test_configs = np.setdiff1d(range(nmol),trainrangetot)
ntest = len(test_configs)
natoms_test = natoms[test_configs]
print("Number of training molecules =", ntrain)
print("Number of testing molecules =", ntest)

# define testing indexes
atomicindx_test = atomicindx[test_configs,:,:]
atom_counting_test = atom_counting[test_configs]

run_prediction(ntest, natmax, natoms_test,
    atom_counting_test, atomicindx_test, test_configs,
    M, elements,
    kernelconfbase,
    basisfilename,
    elselfilebase+str(M)+".txt",
    weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",
    predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

