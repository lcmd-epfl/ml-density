#!/usr/bin/env python3

import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,get_elements_list,get_atomicindx
from power_spectra_lib import read_ps_1mol

conf = Config()

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
refsselfilebase = conf.paths['refs_sel_base']
powerrefbase    = conf.paths['ps_ref_base']
splitpsfilebase = conf.paths['ps_split_base']


(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

# elements array and atomic indices sorted by elements
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

#====================================== reference environments
ref_indices = np.loadtxt(refsselfilebase+str(M)+".txt",int)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)
llmax = max(lmax.values())


ref_indices = list(ref_indices)
ref_imol = [None]*M
ref_iat  = [None]*M
ienv = 0
for imol in range(nmol):
    for iat in range(natoms[imol]):
        if ienv in ref_indices:
             ind = ref_indices.index(ienv)
             ref_imol[ind] = imol
             ref_iat [ind] = iat
        ienv += 1

for l in range(llmax+1):

    print(" l =", l)
    power_training = {}
    for imol in set(ref_imol):
        (nfeat, power_training[imol]) = read_ps_1mol(splitpsfilebase+str(l)+"_"+str(imol)+".npy", nel, atom_counting[imol], atomicindx[imol])

    if l==0:
        power_ref_sparse = np.zeros((M,nfeat),float)
    else:
        power_ref_sparse = np.zeros((M,2*l+1,nfeat),float)

    for ind in range(M):
        imol = ref_imol[ind]
        iat  = ref_iat  [ind]
        power_ref_sparse[ind] = power_training[imol][iat]

    np.save(powerrefbase+str(l)+"_"+str(M)+".npy", power_ref_sparse);

