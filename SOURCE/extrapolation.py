#!/usr/bin/env python3

import numpy as np
from config import Config
from functions import moldata_read,get_elements_list,get_el_list_per_conf,get_atomicindx
from run_prediction import run_prediction

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
xyzexfilename   = conf.paths['ex_xyzfile']
kernelexbase    = conf.paths['ex_kernel_base']
predictfilebase = conf.paths['ex_predict_base']

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
(nmol_ex, natoms_ex, atomic_numbers_ex) = moldata_read(xyzexfilename)
natmax_ex = max(natoms_ex)

# elements array
elements = get_elements_list(atomic_numbers)
elements_ex = get_elements_list(atomic_numbers_ex)
if not set(elements_ex).issubset(set(elements)):
    print("different elements in the molecule and in the training set:", list(elements_ex), "and", list(elements))
    exit(1)

(atom_counting_ex, el_list_per_conf_ex) = get_el_list_per_conf(elements, nmol_ex, natoms_ex, atomic_numbers_ex)
atomicindx_ex = get_atomicindx(nmol_ex, len(elements), natmax_ex, atom_counting_ex, el_list_per_conf_ex)
test_configs = np.arange(nmol_ex)

run_prediction(nmol_ex, natmax_ex, natoms_ex,
    atom_counting_ex, atomicindx_ex, test_configs,
    M, elements,
    kernelexbase,
    basisfilename,
    elselfilebase+str(M)+".txt",
    weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",
    predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

