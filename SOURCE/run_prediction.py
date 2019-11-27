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
    M, species,
    kernelbase,
    basisfilename,
    specselfilename,
    weightsfilename,
    predictfilename):

  # species dictionary, max. angular momenta, number of radial channels
  (spe_dict, lmax, nmax) = basis_read(basisfilename)
  if list(species) != list(spe_dict.values()):
      print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
      exit(1)

  # basis set size
  llmax = max(lmax.values())
  nnmax = max(nmax.values())
  [bsize, almax, anmax] = basis_info(spe_dict, lmax, nmax)

  # reference environments
  fps_species = np.loadtxt(specselfilename, int)

  # sparse kernel sizes
  kernel_sizes = get_kernel_sizes(test_configs, fps_species, spe_dict, M, lmax, atom_counting)

  # regression weights
  weights = np.load(weightsfilename)
  w = unravel_weights(M, llmax, nnmax, fps_species, anmax, almax, weights)

  array_1d_int    = npct.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
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
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_1d_int,
    array_4d_double,
    array_5d_double,
    ctypes.c_char_p ]

  coeffs = np.zeros((nmol, natmax, llmax+1, nnmax, 2*llmax+1))

  ret = prediction.prediction(
      len(species),
      llmax   ,
      nnmax   ,
      M       ,
      nmol    ,
      natmax  ,
      atomicindx.flatten().astype(np.uint32)            ,
      atom_counting.flatten().astype(np.uint32)         ,
      test_configs.astype(np.uint32)                    ,
      natoms.astype(np.uint32)                          ,
      kernel_sizes.astype(np.uint32)                    ,
      fps_species.astype(np.uint32)                     ,
      almax.astype(np.uint32)                           ,
      anmax.flatten().astype(np.uint32)                 ,
      w, coeffs, kernelbase.encode('ascii')             )

  np.save(predictfilename, coeffs)
  return

