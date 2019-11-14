#!/usr/bin/python3

import numpy as np
import prediction
from config import Config
from basis import basis_read
from functions import *

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

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)

(atom_counting_ex, spec_list_per_conf_ex) = get_spec_list_per_conf(species, ndata_ex, natoms_ex, atomic_numbers_ex)

# atomic indices sorted by number
atomicindx_ex = get_atomicindx(ndata_ex, nspecies, natmax_ex, atom_counting_ex, spec_list_per_conf_ex)
atomicindx_ex = atomicindx_ex.T

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

test_configs = np.arange(ndata_ex)
test_species = np.zeros((ndata_ex,natmax_ex),int)
for itest in range(ndata_ex):
    test_species[itest] = spec_list_per_conf_ex[itest]

# sparse kernel sizes
kernel_sizes = get_kernel_sizes(test_configs, fps_species, spe_dict, M, lmax, atom_counting_ex)

# load regression weights
weights = np.load(weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")
w = unravel_weights(M, llmax, nnmax, fps_species, anmax, almax, weights)

# load testing kernels and perform prediction
coeffs = prediction.prediction(kernelexbase,
                               kernel_sizes,fps_species,atom_counting_ex,atomicindx_ex,nspecies,ndata_ex,natmax_ex,
                               llmax,nnmax,natoms_ex,test_configs,test_species,almax,anmax,M,w)
np.save(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",coeffs)

