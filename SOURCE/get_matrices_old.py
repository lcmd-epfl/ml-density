#!/usr/bin/env python3

import sys
import os
import ctypes
import numpy as np
from config import read_config
from basis import basis_read
from functions import moldata_read, get_elements_list, get_atomicindx, basis_info, get_kernel_sizes, nao_for_mol, get_training_sets
import ctypes_def


def main():
    o, p = read_config(sys.argv)

    task = 'b' if (len(sys.argv)>1 and sys.argv[1][0].lower()=='b') else 'a'

    # load molecules
    _, natoms, atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)
    natmax = max(natoms)
    nel = len(elements)

    # training set selection
    nfrac, ntrains, train_configs = get_training_sets(p.trainfilename, o.fracs)
    ntrain = ntrains[-1]
    natoms_train = natoms[train_configs]
    atom_indices, atom_counting, element_indices = get_atomicindx(elements, atomic_numbers[train_configs], natmax)

    # basis set info
    el_dict, lmax, nmax = basis_read(p.basisfilename)
    bsize, alnum, annum = basis_info(el_dict, lmax, nmax);
    llmax = max(lmax.values())
    nnmax = max(nmax.values())

    # problem dimensionality
    ref_elements = np.loadtxt(p.elselfilebase+str(o.M)+".txt",int)
    totsize = sum(bsize[ref_elements])
    ao_sizes = np.array([nao_for_mol(atoms, lmax, nmax) for atoms in atomic_numbers[train_configs]])
    kernel_sizes = get_kernel_sizes(train_configs, ref_elements, el_dict, o.M, lmax, atom_counting)

    # C arguments
    outputfiles = (ctypes.c_char_p * nfrac)()
    for i, frac in enumerate(o.fracs):
        outputfiles[i] = (f'{p.bmatfilebase if task=="b" else p.avecfilebase}_M{o.M}_trainfrac{frac}.dat').encode('ascii')
    qcfilebase = p.goodoverfilebase if task=='b' else p.baselinedwbase

    arguments = ((totsize                           ,      ctypes.c_int,                  ),
                 (nel                               ,      ctypes.c_int,                  ),
                 (llmax                             ,      ctypes.c_int,                  ),
                 (nnmax                             ,      ctypes.c_int,                  ),
                 (o.M                               ,      ctypes.c_int,                  ),
                 (ntrain                            ,      ctypes.c_int,                  ),
                 (natmax                            ,      ctypes.c_int,                  ),
                 (nfrac                             ,      ctypes.c_int,                  ),
                 (ntrains.astype(np.uint32)         ,      ctypes_def.array_1d_int,       ),
                 (atom_indices.astype(np.uint32)    ,      ctypes_def.array_3d_int,       ),
                 (atom_counting.astype(np.uint32)   ,      ctypes_def.array_2d_int,       ),
                 (train_configs.astype(np.uint32)   ,      ctypes_def.array_1d_int,       ),
                 (natoms_train.astype(np.uint32)    ,      ctypes_def.array_1d_int,       ),
                 (ao_sizes.astype(np.uint32)        ,      ctypes_def.array_1d_int,       ),
                 (kernel_sizes.astype(np.uint32)    ,      ctypes_def.array_1d_int,       ),
                 (element_indices.astype(np.uint32) ,      ctypes_def.array_2d_int,       ),
                 (ref_elements.astype(np.uint32)    ,      ctypes_def.array_1d_int,       ),
                 (alnum.astype(np.uint32)           ,      ctypes_def.array_1d_int,       ),
                 (annum.astype(np.uint32)           ,      ctypes_def.array_2d_int,       ),
                 (qcfilebase.encode('ascii')        ,      ctypes.c_char_p,               ),
                 (p.kernelconfbase.encode('ascii')  ,      ctypes.c_char_p,               ),
                 (outputfiles                       ,      ctypes.POINTER(ctypes.c_char_p)))

    args     = [i[0] for i in arguments]
    argtypes = [i[1] for i in arguments]

    get_matrices = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/clibs/get_matrices.so")
    #get_matrices = ctypes.CDLL(os.path.dirname(sys.argv[0])+"/clibs/get_matrices.so", ctypes.RTLD_GLOBAL)
    if task == 'b':
        get_matrices.get_b.restype = ctypes.c_int
        get_matrices.get_b.argtypes = argtypes
        ret = get_matrices.get_b(*args)
    else:
        get_matrices.get_a.restype = ctypes.c_int
        get_matrices.get_a.argtypes = argtypes
        ret = get_matrices.get_a(*args)


if __name__=='__main__':
    main()
