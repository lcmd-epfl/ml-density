#!/usr/bin/env python3

import numpy as np
from basis import basis_read_full
from config import Config
from functions import moldata_read,number_of_electrons_ao

conf = Config()

xyzfilename    = conf.paths['xyzfile']
basisfilename  = conf.paths['basisfile']
trainfilename  = conf.paths['trainingselfile']
qfilebase      = conf.paths['charges_base']

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)
trainrangetot = np.loadtxt(trainfilename, dtype=int)

for imol in trainrangetot:
  q = number_of_electrons_ao(basis, atomic_numbers[imol])
  np.savetxt(qfilebase+str(imol)+".dat", q, fmt='%.10e')
