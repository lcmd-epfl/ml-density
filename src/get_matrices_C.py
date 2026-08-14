#!/usr/bin/env python3
"""Assemble the regression Gram matrix and target vector (C backend, SA-GPR only).

The sparse-GP models (gpr_PITC, gpr_DTC) are not implemented here -- run the Python
get_matrices.py for them. See the guard in main() for the rationale.
"""

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


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['get_gram_matrix'])

    if o.is_gpr:
        # The C backend (get_a/get_b) implements only the SA-GPR formulation.
        #
        # PITC is deliberately Python-only and has no C counterpart: its per-molecule work is a few large
        # dense LAPACK/BLAS operations (the Cholesky/inverse of Lambda_i and the K^T Lambda^-1 K products),
        # so it is already BLAS-bound -- a C port would call the same libraries for no speedup, while
        # forcing us to duplicate and keep in sync the memory-chunked, MPI-parallel reference
        # implementation in libs/gram_matrix.py. Fail fast rather than silently producing an SA-GPR Gram
        # matrix, the wrong operator for gpr_PITC (regression.py expects K^T Lambda^-1 K, not K^T M K).
        #
        # gpr_DTC solves the SA-GPR system, so this backend would build its Gram matrix and target vector
        # correctly -- but it does not write the ml_terms file that regression.py's sigma_p^2 fit reads,
        # so it is excluded too until it does.
        msg = (f'get_matrices_C.py (C backend) does not support regression_model = {o.regression_model} '
               '-- it implements only the SA-GPR formulation and writes no ml_terms file. Run '
               'get_matrices.py instead (with -b for the Gram matrix; gpr_DTC also needs the plain '
               'invocation for the target vector, gpr_PITC builds both together).')
        raise RuntimeError(msg)

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
    nao_ref = basis.nao_for_mol(ref_elements)

    # C arguments
    outputfiles = (ctypes.c_char_p * nfrac)()
    for i, frac in enumerate(o.fracs):
        outputfiles[i] = (p.gram_mat if args.get_gram_matrix else p.target_vec).format(train_frac=frac).encode('ascii')

    kernelfiles = (ctypes.c_char_p * nmol)()
    qcfiles     = (ctypes.c_char_p * nmol)()
    for imol in range(nmol):
        kernelfiles[imol] = p.kernel_nm.format(imol).encode('ascii')
        qcfiles[imol] = (p.metric_matrix if args.get_gram_matrix else p.projection).format(imol).encode('ascii')

    array_1d_int = np.ctypeslib.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
    array_2d_int = np.ctypeslib.ndpointer(dtype=np.uint32,  ndim=2, flags='CONTIGUOUS')

    arguments, argtypes = zip(
            (nao_ref                           ,  ctypes.c_int,                  ),
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
    # the compiled library keeps the historical get_a/get_b symbol names
    func = get_matrices.get_b if args.get_gram_matrix else get_matrices.get_a
    func.restype = ctypes.c_int
    func.argtypes = argtypes
    return func(*arguments)


def basis_info(basis):
    """Build compact basis descriptors required by the C backend.

    Args:
        basis (.functions.Basis): Basis set information.

    Returns:
        tuple[np.ndarray[int], np.ndarray[int]: Arrays containing l counts and 0-padded n-channel counts
            for each element in the ascending order.
    """
    alnum = np.array([basis.lmax[q]+1 for q in basis.elements])
    annum = vstack_padding([basis.nmax[q] for q in basis.elements]).T
    return alnum, annum


def get_atomicindx(elements, atomic_numbers):
    """Count atoms of each element for each molecule.

    Args:
        elements (np.ndarray[int]): Ordered unique atomic numbers.
        atomic_numbers (np.ndarray[int]): Per-molecule atom number arrays.

    Returns:
        np.ndarray[int]: Matrix where atom_counting[imol, iq] is the count of element iq in molecule imol.
    """
    atom_counting = np.zeros((len(atomic_numbers), len(elements)), dtype=int)
    for imol, atoms in enumerate(atomic_numbers):
        for iq, q in enumerate(elements):
            atom_counting[imol, iq] = np.count_nonzero(atoms==q)
    return atom_counting


if __name__=='__main__':
    main()
