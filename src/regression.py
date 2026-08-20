#!/usr/bin/env python3
"""Solve the regression system and save model weights."""

import gc
import numpy as np
import pandas as pd
import scipy.linalg as spl
from numba import jit
import metatensor
from libs.tmap import vector2tmap
from libs.config import get_settings
from libs.config_utils import GPR_DTC, GPR_PITC
from libs.functions import Basis
from libs.kernels_lib import kernel_block_to_dense_self
from libs.gp_common import kmm_cholesky, robust_cholesky
from libs import dtc_lib, pitc_lib
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()

    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, ref_elements)
    nao_ref = basis.nao_for_mol(ref_elements)

    mat  = np.ndarray((nao_ref,nao_ref))

    logger.debug(f'problem dimensionality = {nao_ref}')

    def save_gpr(frac, mat, target_vec, lam, jit):
        """Cholesky-solve one sparse-GP system, then save the factor, the prior scale and the weights.

        Shared by the gpr_DTC and gpr_PITC branches, which hand in the matrix their own equations
        call for: DTC's A = K_TR^T M_T K_TR + lambda K_RR (main.pdf Eq. 21), PITC's Sigma_M. `lam`
        is DTC's lambda, and None for PITC; it selects the model's own prior-scale estimator --
        Eq. 26 for DTC, its counterpart for PITC.

        Args:
            frac (float): Training fraction, formatted into every output path.
            mat (np.ndarray): The system matrix, lower triangle filled. Overwritten.
            target_vec (np.ndarray): The right-hand side.
            lam (float | None): lambda for gpr_DTC, None for gpr_PITC.
            jit (float): Relative jitter robust_cholesky starts from. o.jit for gpr_PITC; 0.0 for
                gpr_DTC, whose jitter is already inside `mat` in SA-GPR's absolute convention.
        """
        L, used_jit = robust_cholesky(mat, jit)
        if used_jit!=jit:
            logger.warning(f'{o.regression_model} system needed relative jitter {used_jit} (bigger than {jit}) to be positive definite')
        x = spl.cho_solve((L, True), target_vec)
        np.save(p.cholesky.format(train_frac=frac), L)

        quad, n_ao = np.loadtxt(p.ml_terms.format(train_frac=frac))
        # Always evaluated, even when the config pins sigma_p^2: it costs one inner product on top
        # of the solve, and it is the number the pinned one overrides, so it belongs in the log.
        if lam is None:
            fitted = pitc_lib.fit_prior_scale(quad, target_vec @ x, n_ao)
        else:
            fitted = dtc_lib.fit_prior_scale(quad, target_vec @ x, n_ao, lam)
        if o.fit_sigma_p2:
            logger.info(f'fitted prior scale sigma_p^2 = {fitted:.6e} (from {int(n_ao)} training AO coefficients)')
            np.savetxt(p.sigma_p2.format(train_frac=frac), [fitted])
        else:
            logger.info(f'using the configured prior scale sigma_p^2 = {o.sigma_p2:.6e} '
                        f'(the type-II maximum-likelihood estimate it overrides is {fitted:.6e})')

        metatensor.save(p.weights.format(train_frac=frac), vector2tmap(ref_elements, basis.llist, x))

    if o.regression_model==GPR_PITC:
        k_MM_tmap = metatensor.load(p.kernel_mm)
        k_mm_dense, _ = kmm_cholesky(basis, ref_elements, k_MM_tmap, o.jit)

        for frac in o.fracs:
            target_vec = np.loadtxt(p.target_vec.format(train_frac=frac))
            mat[:] = 0
            unravel_tril(mat, np.fromfile(p.gram_mat.format(train_frac=frac)), 0.0)
            mat += k_mm_dense
            save_gpr(frac, mat, target_vec, lam=None, jit=o.jit)
    else:
        k_MM = metatensor.load(p.kernel_mm)
        idx = basis.sparse_indices(ref_elements)

        for frac in o.fracs:
            target_vec = np.loadtxt(p.target_vec.format(train_frac=frac))
            gram_file = p.gram_mat.format(train_frac=frac)
            reg = fit_reg(o, p, basis, ref_elements, k_MM, frac, mat, gram_file, target_vec) if o.fit_reg else o.reg
            mat[:] = 0
            # gpr_DTC solves the same system as SA-GPR, so the two share this call unchanged --
            # including o.jit as an *absolute* diagonal addition, unlike the relative jitter the GP
            # models use elsewhere.
            fill_matrix(mat, k_MM, gram_file, idx, basis.nmax, o.jit, reg)
            if o.regression_model==GPR_DTC:
                # robust_cholesky starts at 0 and escalates only if the factorization actually
                # fails, so gpr_DTC and SA-GPR solve one and the same matrix.
                save_gpr(frac, mat, target_vec, lam=reg, jit=0.0)
                continue
            weights = spl.solve(mat, target_vec, assume_a='sym', lower=True, overwrite_a=True, overwrite_b=True)
            weights = vector2tmap(ref_elements, basis.llist, weights)
            metatensor.save(p.weights.format(train_frac=frac), weights)


def fit_reg(o, p, basis, ref_elements, k_MM, frac, mat, gram_file, target_vec):
    """Pick eta by marginal likelihood for one training fraction, and persist it.

    Only reachable with `regularisation = fit`, which config_paths.py restricts to gpr_DTC.
    `mat` is loaded with the Gram matrix alone here, because dtc_lib.fit_regularisation has to move
    lambda around on the K_RR term; the caller then rebuilds the system matrix through fill_matrix at
    the fitted lambda,
    so the factorized system is bit-for-bit the one a fixed-eta run would produce.

    Peak memory is three P x P matrices (the Gram, K_MM, and one Cholesky factor) against the two a
    fixed-eta run needs.

    Args:
        o (SimpleNamespace): Configured options.
        p (SimpleNamespace): Configured paths.
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elements (np.ndarray[int]): Reference-environment atomic numbers.
        k_MM (metatensor.TensorMap): Reference-reference kernel.
        frac (float): Training fraction, formatted into every path.
        mat (np.ndarray): The (P, P) scratch matrix. Overwritten.
        gram_file (str): Path to the packed Gram-matrix binary file.
        target_vec (np.ndarray): The right-hand side.

    Returns:
        float: The fitted eta.
    """
    quad, n_ao = np.loadtxt(p.ml_terms.format(train_frac=frac))
    mat[:] = 0
    data = np.fromfile(gram_file)
    unravel_tril(mat, data, o.jit)
    del data
    gc.collect()
    k_mm_dense = np.tril(kernel_block_to_dense_self(basis, ref_elements, k_MM))
    reg = dtc_lib.fit_regularisation(mat, k_mm_dense, target_vec, quad, n_ao)
    del k_mm_dense
    gc.collect()
    logger.info(f'fitted regularisation lambda = {reg:.6e} (marginal likelihood, {int(n_ao)} training AO coefficients)')
    np.savetxt(p.fitted_reg.format(train_frac=frac), [reg])
    return reg


@jit(nopython=True)
def unravel_tril(mat, data, jitter):
    """Fill a lower-triangular matrix from its vector form and add a constant to the diagonal.

    One should not use `np.tril_indices()` because it creates a huge indices array.

    Args:
        mat (np.ndarray): Updated in place.
        data (np.ndarray): Packed lower-triangular coefficients.
        jitter (float): Diagonal jitter added for numerical stability.
    """
    n = mat.shape[0]
    k = 0
    for j in range(n):
        for i in range(j+1):
            mat[j,i] = data[k]
            k += 1
        mat[j,j] += jitter


def fill_matrix(mat, k_MM, gram_file, idx, nmax, jitter, reg):
    """Assemble the regression matrix from the Gram matrix and add regularization.

    Args:
        mat (np.ndarray): Vector representation of the lower triangle of a symmetric matrix.
        k_MM (metatensor.TensorMap): Reference-reference kernel.
        gram_file (str): Path to packed Gram-matrix binary file.
        idx (np.ndarray): Sparse AO start indices per reference and l.
        nmax (dict[int, np.ndarray]): Radial basis sizes indexed by element and l.
        jitter (float): Diagonal jitter added after unpacking the Gram matrix.
        reg (float): Regularization coefficient scaling kernel blocks.
    """
    data = np.fromfile(gram_file)
    unravel_tril(mat, data, jitter)
    del data
    gc.collect()
    for (l, q), kblock in k_MM.items():
        msize = 2*l+1
        for iiref12, (iref1, iref2) in enumerate(kblock.samples):
            if iref1<iref2:
                continue
            dk = reg * kblock.values[iiref12,:,:,0]
            for n in range(nmax[q][l]):
                i1 = idx[iref1, l] + n*msize
                i2 = idx[iref2, l] + n*msize
                mat[i1:i1+msize, i2:i2+msize] += dk


if __name__=='__main__':
    main()
