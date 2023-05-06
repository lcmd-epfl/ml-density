#!/usr/bin/env python3

import sys
import os
import ctypes
import numpy as np
from config import read_config
from basis import basis_read
from functions import moldata_read, get_elements_list, nao_for_mol, get_training_sets
import ctypes_def


def main():
    o, p = read_config(sys.argv)

    task = 'b' if (len(sys.argv)>1 and sys.argv[1][0].lower()=='b') else 'a'

    # load molecules
    _, natoms, atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)
    natmax = max(natoms)
    nel = len(elements)

    # reference environments
    ref_elements = np.loadtxt(f'{p.qrefsselfilebase}{o.M}.txt', dtype=int)
    ref_elements_idx = np.zeros_like(ref_elements)
    for iq, q in enumerate(elements):
        ref_elements_idx[np.where(ref_elements==q)] = iq

    # training set selection
    nfrac, ntrains, train_configs = get_training_sets(p.trainfilename, o.fracs)
    ntrain = ntrains[-1]
    natoms_train = natoms[train_configs]
    atom_indices, atom_counting, element_indices = get_atomicindx(elements, atomic_numbers[train_configs], natmax)

    # basis set info
    lmax, nmax = basis_read(p.basisfilename)
    bsize, alnum, annum = basis_info(elements, lmax, nmax);
    llmax = max(lmax.values())
    nnmax = max(nmax.values())

    # problem dimensionality
    totsize = sum(bsize[ref_elements_idx])
    ao_sizes = np.array([nao_for_mol(atoms, lmax, nmax) for atoms in atomic_numbers[train_configs]])
    kernel_sizes = get_kernel_sizes(elements, ref_elements_idx, lmax, atom_counting)

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
                 (ref_elements_idx.astype(np.uint32),      ctypes_def.array_1d_int,       ),
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


def basis_info(elements, lmax, nmax):
    nel = len(elements)
    llmax = max(lmax.values())
    bsize = np.zeros(nel, dtype=int)
    alnum = np.zeros(nel, dtype=int)
    annum = np.zeros((nel, llmax+1), dtype=int)
    for iel, q in enumerate(elements):
        alnum[iel] = lmax[q]+1
        for l in range(lmax[q]+1):
            annum[iel,l] = nmax[(q,l)]
            bsize[iel] += nmax[(q,l)]*(2*l+1)
    return [bsize, alnum, annum]


def get_kernel_sizes(elements, ref_elements_idx, lmax, atom_counting):
    size_k = np.array([sum((2*l+1)**2 for l in range(lmax[q]+1)) for q in elements])
    kernel_sizes = atom_counting[:, ref_elements_idx] @ size_k[ref_elements_idx]
    return kernel_sizes


def get_atomicindx(elements, atomic_numbers, natmax):
    '''
    element_indices[imol, :]   for each atom its element number iel
    atom_counting[imol, iel]   number of atoms of element #iel
    atom_indices[imol, iel, :] indices of atoms of element #iel
    '''
    nmol = len(atomic_numbers)
    nel  = len(elements)
    atom_counting   = np.zeros((nmol, nel), dtype=int)
    atom_indices    = np.zeros((nmol, nel, natmax), dtype=int)
    element_indices = np.zeros((nmol, natmax), dtype=int)
    for imol, atoms in enumerate(atomic_numbers):
        for iel, el in enumerate(elements):
            idx = np.where(atomic_numbers[imol]==el)[0]
            count = len(idx)
            atom_indices[imol, iel, 0:count] = idx
            atom_counting[imol, iel] = count
            element_indices[imol, idx] = iel
    return atom_indices, atom_counting, element_indices


if __name__=='__main__':
    main()
