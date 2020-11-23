#!/usr/bin/env python3

import numpy as np
from basis import basis_read_full
from config import Config
from functions import number_of_electrons_ao,moldata_read,get_elements_list

conf = Config()

def set_variable_values():
    f  = conf.get_option('trainfrac'   ,  1.0,   float)
    return [f]

[frac] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
qfilebase       = 'qvec/mol'

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
elements = get_elements_list(atomic_numbers)

(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)

trainfilename    = conf.paths['trainingselfile']
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
train_configs = np.array(sorted(trainrangetot[0:ntrain]))

for imol in train_configs:
    atoms = atomic_numbers[imol]
    q = number_of_electrons_ao(basis, atoms)
    np.savetxt(qfilebase+str(imol)+".dat", q, fmt='%.10e')
