#!/usr/bin/python3

import numpy as np
from basis import basis_read
from config import Config
from functions import *
import os
import sys
import ctypes
import numpy.ctypeslib as npct

conf = Config()

def set_variable_values():
    f  = conf.get_option('trainfrac'   ,  1.0,   float)
    m  = conf.get_option('m'           ,  100,   int  )
    r  = conf.get_option('regular'     ,  1e-6,  float)
    j  = conf.get_option('jitter'      ,  1e-10, float)
    return [f,m,r,j]

[frac,M,reg,jit] = set_variable_values()

kmmbase         = conf.paths['kmm_base']
avecfilebase    = conf.paths['avec_base']
bmatfilebase    = conf.paths['bmat_base']
weightsfilebase = conf.paths['weights_base']
xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
specselfilebase = conf.paths['spec_sel_base']

#====================================== reference environments
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
species = get_species_list(atomic_numbers)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)

# basis set size
llmax = max(lmax.values())
[bsize, almax, anmax] = basis_info(spe_dict, lmax, nmax);
totsize = sum(bsize[fps_species])

#===============================================================

k_MM = np.load(kmmbase+str(M)+".npy")
Avec = np.loadtxt(avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt")
mat  = np.zeros((totsize,totsize))

array_1d_int    = npct.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')
array_3d_double = npct.ndpointer(dtype=np.float64, ndim=3, flags='CONTIGUOUS')
regression = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/regression.so")
regression.make_matrix.restype = ctypes.c_int
regression.make_matrix.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  array_1d_int,
  array_1d_int,
  array_1d_int,
  array_3d_double,
  array_2d_double,
  ctypes.c_double,
  ctypes.c_double,
  ctypes.c_char_p ]

ret = regression.make_matrix(
      totsize,
      llmax  ,
      M      ,
      fps_species.astype(np.uint32),
      almax.astype(np.uint32),
      anmax.flatten().astype(np.uint32),
      k_MM, mat, reg, jit,
      (bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".dat").encode('ascii'))

print("problem dimensionality =", totsize)

weights = np.linalg.solve(mat, Avec)
np.save(weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",weights)

