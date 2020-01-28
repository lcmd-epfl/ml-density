#!/usr/bin/env python3

import numpy as np
from basis import basis_read
from config import Config
from functions import moldata_read,get_elements_list,basis_info
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
elselfilebase   = conf.paths['spec_sel_base']

kmmfile     = kmmbase+str(M)+".npy"
avecfile    = avecfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".txt"
bmatfile    = bmatfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".dat"
elselfile   = elselfilebase+str(M)+".txt"
weightsfile = weightsfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy"

# print ("  input:")
# print ( xyzfilename )
# print ( basisfilename)
# print ( elselfile )
# print ( kmmfile )
# print ( avecfile )
# print ( bmatfile )
# print ("  output:")
# print (weightsfile)
# print ()

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

#===============================================================

k_MM = np.load(kmmfile)
Avec = np.loadtxt(avecfile)
mat  = np.zeros((totsize,totsize))

array_1d_int    = npct.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
array_2d_int    = npct.ndpointer(dtype=np.uint32,  ndim=2, flags='CONTIGUOUS')
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
  array_2d_int,
  array_3d_double,
  array_2d_double,
  ctypes.c_double,
  ctypes.c_double,
  ctypes.c_char_p ]

ret = regression.make_matrix(
      totsize,
      llmax  ,
      M      ,
      ref_elements.astype(np.uint32),
      alnum.astype(np.uint32),
      annum.astype(np.uint32),
      k_MM, mat, reg, jit,
      bmatfile.encode('ascii'))

print("problem dimensionality =", totsize)

weights = np.linalg.solve(mat, Avec)
np.save(weightsfile, weights)

