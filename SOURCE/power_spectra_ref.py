#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read, get_elements_list
from power_spectra_lib import merge_ref_ps, get_ref_idx
import equistore

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
refsselfilebase = conf.paths['refs_sel_base']
powerrefbase    = conf.paths['ps_ref_base']
splitpsfilebase = conf.paths['ps_split_base']


_, natoms, atomic_numbers = moldata_read(xyzfilename)
elements = get_elements_list(atomic_numbers)
ref_indices = np.loadtxt(refsselfilebase+str(M)+".txt", dtype=int)

(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)

ref_imol, ref_iat = get_ref_idx(natoms, ref_indices)
tensor = merge_ref_ps(ref_indices, lmax, elements, atomic_numbers, np.vstack((ref_imol, ref_iat)).T, splitpsfilebase+'_{mol_id}.npz')
equistore.save(f'{powerrefbase}_{M}.npz', tensor)
