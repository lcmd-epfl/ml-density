#!/usr/bin/env python3

import sys
import os
import gc
import ctypes
import numpy as np
import scipy.linalg as spl
from basis import basis_read
from config import read_config
from functions import moldata_read, get_elements_list, basis_info
import ctypes_def


def main():
    o, p = read_config(sys.argv)

    kmmfile   = f'{p.kmmbase}{o.M}.npy'
    elselfile = f'{p.elselfilebase}{o.M}.txt'
    ref_elements = np.loadtxt(elselfile, int)

    nmol, natoms, atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)

    # elements dictionary, max. angular momenta, number of radial channels
    (el_dict, lmax, nmax) = basis_read(p.basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    # basis set size
    llmax = max(lmax.values())
    [bsize, alnum, annum] = basis_info(el_dict, lmax, nmax);
    totsize = sum(bsize[ref_elements])
    print(f'problem dimensionality = {totsize}')

    k_MM = np.load(kmmfile)
    mat  = np.ndarray((totsize,totsize))

    regression = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/clibs/regression.so")
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

    for frac in o.fracs:
      avecfile    = f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt'
      bmatfile    = f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat'
      weightsfile = f'{p.weightsfilebase}_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
      Avec = np.loadtxt(avecfile)
      mat[:] = 0

      ret = regression.make_matrix(
            totsize,
            llmax  ,
            o.M    ,
            ref_elements.astype(np.uint32),
            alnum.astype(np.uint32),
            annum.astype(np.uint32),
            k_MM, mat, o.reg, o.jit,
            bmatfile.encode('ascii'))
      print(ret)

      weights = spl.solve(mat, Avec, assume_a='sym', overwrite_a=True, overwrite_b=True)
      np.save(weightsfile, weights)
      del Avec
      del weights
      gc.collect()


if __name__=='__main__':
    main()
