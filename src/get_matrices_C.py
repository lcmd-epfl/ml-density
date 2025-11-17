#!/usr/bin/env python3

import sys
import os
import ctypes
import numpy as np
from libs.config import read_config
from libs.basis import basis_read
from libs.functions import moldata_read, get_elements_list, get_training_sets


def main():
    o, p = read_config(sys.argv)

    task = 'b' if (len(sys.argv)>1 and sys.argv[1][0].lower()=='b') else 'a'

    # load molecules
    atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)

    # reference environments
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]
    ref_elements_idx = np.zeros_like(ref_elements)
    for iq, q in enumerate(elements):
        ref_elements_idx[np.where(ref_elements==q)] = iq

    # training set selection
    nfrac, ntrains, train_configs = get_training_sets(p.trainfilename, o.fracs)
    ntrain = ntrains[-1]
    atomic_numbers_train = atomic_numbers[train_configs]
    atom_counting = get_atomicindx(elements, atomic_numbers_train)

    # basis set info
    lmax, nmax = basis_read(p.basisfilename)
    bsize, alnum, annum = basis_info(elements, lmax, nmax)

    # problem dimensionality
    totsize = sum(bsize[ref_elements_idx])

    # C arguments
    outputfiles = (ctypes.c_char_p * nfrac)()
    for i, frac in enumerate(o.fracs):
        if task=="b":
            outputfiles[i] = (f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat').encode('ascii')
        else:
            outputfiles[i] = (f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt').encode('ascii')
    qcfilebase = p.goodoverfilebase if task=='b' else p.baselinedwbase

    array_1d_int = np.ctypeslib.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
    array_2d_int = np.ctypeslib.ndpointer(dtype=np.uint32,  ndim=2, flags='CONTIGUOUS')

    arguments = ((totsize                           ,      ctypes.c_int,                  ),
                 (len(elements)                     ,      ctypes.c_int,                  ),
                 (o.M                               ,      ctypes.c_int,                  ),
                 (ntrain                            ,      ctypes.c_int,                  ),
                 (nfrac                             ,      ctypes.c_int,                  ),
                 (ntrains.astype(np.uint32)         ,      array_1d_int,                  ),
                 (atom_counting.astype(np.uint32)   ,      array_2d_int,                  ),
                 (train_configs.astype(np.uint32)   ,      array_1d_int,                  ),
                 (ref_elements_idx.astype(np.uint32),      array_1d_int,                  ),
                 (alnum.astype(np.uint32)           ,      array_1d_int,                  ),
                 (annum.flatten().astype(np.uint32) ,      array_1d_int,                  ),
                 (elements.astype(np.uint32)        ,      array_1d_int,                  ),
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
    return ret


def basis_info(elements, lmax, nmax):
    nel = len(elements)
    llmax = max(lmax.values())
    bsize = np.zeros(nel, dtype=int)
    alnum = np.zeros(nel, dtype=int)
    annum = np.zeros((llmax+1, nel), dtype=int)
    for iq, q in enumerate(elements):
        alnum[iq] = lmax[q]+1
        for l in range(lmax[q]+1):
            annum[l,iq] = nmax[(q,l)]
            bsize[iq]  += nmax[(q,l)]*(2*l+1)
    return bsize, alnum, annum


def get_atomicindx(elements, atomic_numbers):
    '''
    Returns:
      atom_counting[imol, iq]   number of atoms of element #iq in mol #imol
    '''
    atom_counting = np.zeros((len(atomic_numbers), len(elements)), dtype=int)
    for imol, atoms in enumerate(atomic_numbers):
        for iq, q in enumerate(elements):
            atom_counting[imol, iq] = np.count_nonzero(atoms==q)
    return atom_counting


if __name__=='__main__':
    main()
