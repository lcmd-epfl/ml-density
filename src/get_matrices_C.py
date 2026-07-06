#!/usr/bin/env python3

import sys
import os
import ctypes
import numpy as np
import pandas as pd
from qstack.mathutils.array import vstack_padding
from libs.config import get_settings
from libs.functions import moldata_read, get_elements, Basis, Subset
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():
    args, o, p = get_settings(return_args=['get_b_matrix'])

    # load molecules
    atomic_numbers = moldata_read(p.xyzfilename)
    nmol = len(atomic_numbers)
    elements = get_elements(atomic_numbers)

    # reference environments
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    ref_elements_idx = np.zeros_like(ref_elements)
    for iq, q in enumerate(elements):
        ref_elements_idx[np.where(ref_elements==q)] = iq

    # training set selection
    ntrains, train_configs = Subset(p.train_test_sets).get_training_all(o.fracs)
    nfrac = len(ntrains)
    ntrain = ntrains[-1]
    atomic_numbers_train = atomic_numbers[train_configs]
    atom_counting = get_atomicindx(elements, atomic_numbers_train)

    # basis set info
    basis = Basis(o.basisname, elements=ref_elements)
    alnum, annum = basis_info(basis)

    # problem dimensionality
    totsize = basis.nao_for_mol(ref_elements)

    # C arguments
    outputfiles = (ctypes.c_char_p * nfrac)()
    for i, frac in enumerate(o.fracs):
        outputfiles[i] = (p.bmat if args.get_b_matrix else p.avec).format(train_frac=frac).encode('ascii')

    kernelfiles = (ctypes.c_char_p * nmol)()
    qcfiles     = (ctypes.c_char_p * nmol)()
    for imol in range(nmol):
        kernelfiles[imol] = p.kernel_nm.format(imol).encode('ascii')
        qcfiles[imol] = (p.metric_matrix if args.get_b_matrix else p.projection).format(imol).encode('ascii')

    array_1d_int = np.ctypeslib.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
    array_2d_int = np.ctypeslib.ndpointer(dtype=np.uint32,  ndim=2, flags='CONTIGUOUS')

    arguments, argtypes = zip(
            (totsize                           ,  ctypes.c_int,                  ),
            (len(elements)                     ,  ctypes.c_int,                  ),
            (o.M                               ,  ctypes.c_int,                  ),
            (ntrain                            ,  ctypes.c_int,                  ),
            (nfrac                             ,  ctypes.c_int,                  ),
            (ntrains.astype(np.uint32)         ,  array_1d_int,                  ),
            (atom_counting.astype(np.uint32)   ,  array_2d_int,                  ),
            (train_configs.astype(np.uint32)   ,  array_1d_int,                  ),
            (ref_elements_idx.astype(np.uint32),  array_1d_int,                  ),
            (alnum.astype(np.uint32)           ,  array_1d_int,                  ),
            (annum.flatten().astype(np.uint32) ,  array_1d_int,                  ),
            (elements.astype(np.uint32)        ,  array_1d_int,                  ),
            (qcfiles                           ,  ctypes.POINTER(ctypes.c_char_p)),
            (kernelfiles                       ,  ctypes.POINTER(ctypes.c_char_p)),
            (outputfiles                       ,  ctypes.POINTER(ctypes.c_char_p)),
            strict=True)

    get_matrices = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/clibs/get_matrices.so")
    func = get_matrices.get_b if args.get_b_matrix else get_matrices.get_a
    func.restype = ctypes.c_int
    func.argtypes = argtypes
    ret = func(*arguments)
    return ret


def basis_info(basis):
    # basis.elements is ordered
    alnum = np.array([basis.lmax[q]+1 for q in basis.elements])
    annum = vstack_padding([basis.nmax[q] for q in basis.elements]).T
    return alnum, annum


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
