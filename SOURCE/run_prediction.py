import numpy as np
from basis import basis_read
from functions import basis_info,get_kernel_sizes,unravel_weights
import os
import sys
import ctypes
import numpy.ctypeslib as npct

def run_prediction(
    nmol, natmax, natoms,
    atom_counting, atomicindx,
    test_configs,
    M, elements,
    kernelbase,
    basisfilename,
    elselfilename,
    weightsfilename,
    predictfilename):

  # elements dictionary, max. angular momenta, number of radial channels
  (el_dict, lmax, nmax) = basis_read(basisfilename)
  if list(elements) != list(el_dict.values()):
      print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
      exit(1)

  # basis set size
  llmax = max(lmax.values())
  nnmax = max(nmax.values())
  [bsize, alnum, annum] = basis_info(el_dict, lmax, nmax)

  # reference environments
  ref_elements = np.loadtxt(elselfilename, int)

  # sparse kernel sizes
  kernel_sizes = get_kernel_sizes(test_configs, ref_elements, el_dict, M, lmax, atom_counting)

  # regression weights
  weights = np.load(weightsfilename)
  w = unravel_weights(M, llmax, nnmax, ref_elements, annum, alnum, weights)

  array_1d_int    = npct.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
  array_2d_int    = npct.ndpointer(dtype=np.uint32,  ndim=2, flags='CONTIGUOUS')
  array_3d_int    = npct.ndpointer(dtype=np.uint32,  ndim=3, flags='CONTIGUOUS')
  array_4d_double = npct.ndpointer(dtype=np.float64, ndim=4, flags='CONTIGUOUS')
  array_5d_double = npct.ndpointer(dtype=np.float64, ndim=5, flags='CONTIGUOUS')

  prediction = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/prediction.so")
  prediction.prediction.restype = ctypes.c_int
  prediction.prediction.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    array_3d_int,
    array_2d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_2d_int,
    array_4d_double,
    array_5d_double,
    ctypes.c_char_p ]

  coeffs = np.zeros((nmol, natmax, llmax+1, nnmax, 2*llmax+1))

  ret = prediction.prediction(
      len(elements),
      llmax   ,
      nnmax   ,
      M       ,
      nmol    ,
      natmax  ,
      atomicindx.astype(np.uint32)          ,
      atom_counting.astype(np.uint32)       ,
      test_configs.astype(np.uint32)        ,
      natoms.astype(np.uint32)              ,
      kernel_sizes.astype(np.uint32)        ,
      ref_elements.astype(np.uint32)        ,
      alnum.astype(np.uint32)               ,
      annum.astype(np.uint32)               ,
      w, coeffs, kernelbase.encode('ascii') )

  np.save(predictfilename, coeffs)
  return coeffs

