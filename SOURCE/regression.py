#!/usr/bin/env python3

import sys
import numpy as np
import scipy.linalg as spl
from basis import basis_read
from config import Config,get_config_path
from functions import moldata_read,get_elements_list,basis_info
import os
import gc
import ctypes
import ctypes_def

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    f  = conf.get_option('trainfrac', np.array([1.0]), conf.floats)
    m  = conf.get_option('m'        , 100,             int  )
    r  = conf.get_option('regular'  , 1e-6,            float)
    j  = conf.get_option('jitter'   , 1e-10,           float)
    return [f,m,r,j]

[fracs,M,reg,jit] = set_variable_values()

EPSILON         = 5E-13
kmmbase         = conf.paths['kmm_base']
avecfilebase    = conf.paths['avec_base']
bmatfilebase    = conf.paths['bmat_base']
weightsfilebase = conf.paths['weights_base']
xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
elselfilebase   = conf.paths['spec_sel_base']

kmmfile     = kmmbase+str(M)+".npy"
elselfile   = elselfilebase+str(M)+".txt"

#====================================== reference environments
ref_elements = np.loadtxt(elselfile, int)

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
elements = get_elements_list(atomic_numbers)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)

# basis set size
llmax = max(lmax.values())
[bsize, alnum, annum] = basis_info(el_dict, lmax, nmax);
totsize = sum(bsize[ref_elements])
print("problem dimensionality =", totsize)

k_MM = np.load(kmmfile)

regression = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/regression.so")
regression.make_matrix.restype = ctypes.c_int
regression.make_matrix.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_2d_int,
  ctypes_def.array_3d_double,
  ctypes_def.array_2d_double,
  ctypes.c_double,
  ctypes.c_double,
  ctypes.c_char_p ]


mat  = np.ndarray((totsize,totsize))

for frac in fracs:

  avecfile    = avecfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".txt"
  bmatfile    = bmatfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".dat"
  weightsfile = weightsfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy"

  Avec = np.loadtxt(avecfile)

  mat[:] = 0
  
  ret = regression.make_matrix(
        totsize,
        llmax  ,
        M      ,
        ref_elements.astype(np.uint32),
        alnum.astype(np.uint32),
        annum.astype(np.uint32),
        k_MM, mat, reg, jit,
        bmatfile.encode('ascii'))
  print(ret)

  #weights = spl.solve(mat, Avec, assume_a='sym', overwrite_a=True, overwrite_b=True)

  # long way to solve the equation, in a way that avoids numerical instabilities
  diag,proj = spl.eigh(mat)
  print('EIGEN:', diag)
  accepted_values = abs(diag) > EPSILON
  diag2 = diag[accepted_values]
  proj2 = proj.T[accepted_values, :]
  Y2 = proj2 @ Avec
  W2 = Y2 / diag2
  weights = proj2.T @ W2
  remainder = mat @ weights - Avec
  print("closeness:", np.sqrt(remainder@remainder))

  del Avec
  np.save(weightsfile, weights)
  del weights
  gc.collect()


