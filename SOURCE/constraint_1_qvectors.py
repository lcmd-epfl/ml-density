#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from basis import basis_read_full
from functions import moldata_read,number_of_electrons_ao

path = get_config_path(sys.argv)
conf = Config(config_path=path)

xyzfilename    = conf.paths['xyzfile']
basisfilename  = conf.paths['basisfile']
trainfilename  = conf.paths['trainingselfile']
qfilebase      = conf.paths['charges_base']

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)
train_configs = np.loadtxt(trainfilename, dtype=int)

for imol in train_configs:
  q = number_of_electrons_ao(basis, atomic_numbers[imol])
  np.savetxt(qfilebase+str(imol)+".dat", q, fmt='%.10e')

