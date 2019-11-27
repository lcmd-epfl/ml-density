import numpy as np
import prediction
from functions import basis_info,get_kernel_sizes,unravel_weights
from basis import basis_read

def run_prediction(
    nmol, natmax, natoms,
    atom_counting, atomicindx,
    test_configs, test_species,
    M, species,
    kernelexbase,
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

  coeffs = prediction.prediction(
    kernelexbase,
    kernel_sizes,
    fps_species,
    atom_counting,
    atomicindx,
    len(species),
    nmol,
    natmax,
    llmax,
    nnmax,
    natoms,
    test_configs,
    test_species,
    almax,
    anmax,
    M,
    w)

  np.save(predictfilename, coeffs)
  return

