#!/usr/bin/python3

import numpy as np
from config import Config
from functions import moldata_read,get_species_list,get_spec_list_per_conf,get_atomicindx
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
specselfilebase = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
weightsfilebase = conf.paths['weights_base']
xyzexfilename   = conf.paths['ex_xyzfile']
kernelexbase    = conf.paths['ex_kernel_base']
predictfilebase = conf.paths['ex_predict_base']

(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
(ndata_ex, natoms_ex, atomic_numbers_ex) = moldata_read(xyzexfilename)
natmax_ex = max(natoms_ex)

# species array
species = get_species_list(atomic_numbers)
species_ex = get_species_list(atomic_numbers_ex)
if not set(species_ex).issubset(set(species)):
    print("different elements in the molecule and in the training set:", list(species_ex), "and", list(species))
    exit(1)

(atom_counting_ex, spec_list_per_conf_ex) = get_spec_list_per_conf(species, ndata_ex, natoms_ex, atomic_numbers_ex)

# atomic indices sorted by number
atomicindx_ex = get_atomicindx(ndata_ex, len(species), natmax_ex, atom_counting_ex, spec_list_per_conf_ex)
atomicindx_ex = atomicindx_ex.T

test_configs = np.arange(ndata_ex)
test_species = np.zeros((ndata_ex,natmax_ex),int)
for itest in range(ndata_ex):
    test_species[itest] = spec_list_per_conf_ex[itest]

run_prediction(ndata_ex, natmax_ex, natoms_ex,
    atom_counting_ex, atomicindx_ex,
    test_configs, test_species,
    M, species,
    kernelexbase,
    basisfilename,
    specselfilebase+str(M)+".txt",
    weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",
    predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

