#!/usr/bin/env python3

import sys
import numpy as np
from basis import basis_read
from config import Config,get_config_path
from functions import moldata_read,get_elements_list
from kernels_lib import kernel_mm

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
kmmbase         = conf.paths['kmm_base']
powerrefbase    = conf.paths['ps_ref_base']

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
elements = get_elements_list(atomic_numbers)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)
llmax = max(lmax.values())

k_MM = kernel_mm(M, llmax, powerrefbase)

np.save(kmmbase+str(M)+".npy", k_MM )

