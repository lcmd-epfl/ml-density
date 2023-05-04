import numpy as np
from basis import basis_read
from functions import basis_info,get_kernel_sizes,unravel_weights, nao_for_mol, print_progress
import os
import sys
import ctypes
import ctypes_def
from libs.tmap import vector2tmap, tmap2vector, join
import equistore


def compute_prediction(atoms, lmax, nmax, kernel, weights, averages=None):
    nao = nao_for_mol(atoms, lmax, nmax)
    coeffs = vector2tmap(atoms, lmax, nmax, np.zeros(nao))
    for (l, q), cblock in coeffs:
        wblock = weights.block(spherical_harmonics_l=l, species_center=q)
        kblock = kernel.block(spherical_harmonics_l=l, species_center=q)
        for sample in cblock.samples:
            cpos = cblock.samples.position(sample)
            kpos = kblock.samples.position(sample)
            cblock.values[cpos,:,:] = np.einsum('mMr,rMn->mn', kblock.values[kpos], wblock.values)
        if averages and l==0:
            cblock.values[:,:,:] = cblock.values + averages.block(element=q).values
    return coeffs


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

  prediction = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/clibs/prediction.so")
  prediction.prediction.restype = ctypes.c_int
  prediction.prediction.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes_def.array_3d_int,
    ctypes_def.array_2d_int,
    ctypes_def.array_1d_int,
    ctypes_def.array_1d_int,
    ctypes_def.array_1d_int,
    ctypes_def.array_1d_int,
    ctypes_def.array_1d_int,
    ctypes_def.array_2d_int,
    ctypes_def.array_4d_double,
    ctypes_def.array_5d_double,
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



def run_prediction_new(test_configs, atomic_numbers, ref_elements,
                       basisfilename, weightsfilename, kernelbase, predictfilename):

  _, lmax, nmax = basis_read(basisfilename)
  weights = vector2tmap(ref_elements, lmax, nmax, np.load(weightsfilename))

  predictions = []
  for i, (imol, atoms) in enumerate(zip(test_configs, atomic_numbers)):
      print_progress(i, len(test_configs))
      kernel = equistore.load(f'{kernelbase}{imol}.dat.npz')
      predictions.append(compute_prediction(atoms, lmax, nmax, kernel, weights))

  predictions = join(predictions)
  equistore.save(predictfilename, predictions)
