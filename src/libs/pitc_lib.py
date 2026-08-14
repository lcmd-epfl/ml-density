"""PITC (Partially Independent Training Conditional) sparsification helpers.

Shared by the sparse-GP (gpr_PITC/gpr_DTC) branches of target_vector.py/gram_matrix.py/regression.py to
build, per training molecule i, the precision matrix Lambda_i = D_i + eta*metric_i^-1, with
D_i = K_{I_i,I_i} - K_{I_i,M} K_MM^-1 K_{M,I_i} (all three matrices are block-diagonal per molecule),
and the shared K_MM Cholesky factor both target_vector.py and gram_matrix.py need.
"""

import logging
import numpy as np
import scipy.linalg as spl
import metatensor
from qstack.io.metatensor import tensormap_to_array
from libs.kernels_lib import kernel_block_to_dense_rect, kernel_block_to_dense_self, kernel_mm
from libs.tmap import merge_ref_ps
from libs.functions import make_dummy_mol

logger = logging.getLogger('__main__')

def fit_sigma_f2(quad, t_dot_x, n_ao, lam=1.0):
    """Closed-form maximum-likelihood estimate of the kernel amplitude sigma_f^2 (the prior scale).

    A GP kernel factors as k(x,x') = sigma_f^2 r(x,x') with r(x,x) = 1. kernels_lib builds the
    normalized r (lambda-SOAP).

        log p(y) = -1/2 sigma_f^-2 y^T C_1^-1 y - N/2 log sigma_f^2 - 1/2 log|C_1| - N/2 log 2pi

    log p(y) reaches a unique maximum at sigma_f^2 = y^T C_1^-1 y / N. Woodbury on
    C_1 = Lambda + K_NM K_MM^-1 K_MN reduces the quadratic form to
    y^T C_1^-1 y = sum_i y_i^T Lambda_i^-1 y_i - t^T Sigma_M^-1 t, which t the target vector and
    Sigma_M the PITC marginal covariance that are already solved.

    `lam` exists because the two GP models scale that identity differently. PITC solves
    Sigma_M = K_MM + sum_i K^T Lambda_i^-1 K directly and passes both terms in that scaling, so
    lam = 1. gpr_DTC instead solves the SA-GPR matrix A = sum_i K^T M_i K + eta*K_MM = eta*Sigma_M, and
    supplies the two terms unscaled by eta -- sum_i y_i^T M_i y_i (from p.coef_norms) and t^T A^-1 t
    -- each of which is eta times its Sigma_M-scaled counterpart. Passing lam = eta divides that
    factor back out.

    Args:
        quad (float): sum_i y_i^T Lambda_i^-1 y_i, accumulated by do_work_target_pitc (PITC) or
            summed from p.coef_norms (DTC, where it is sum_i y_i^T M_i y_i).
        t_dot_x (float): t^T Sigma_M^-1 t, i.e. the target vector dotted with the solved weights.
        n_ao (float): N, the total number of training AO coefficients.
        lam (float): Scaling of the solved matrix relative to Sigma_M: 1.0 for PITC, o.reg for DTC.

    Returns:
        float: The fitted sigma_f^2.

    Raises:
        RuntimeError: The quadratic form came out non-positive, which is impossible for a positive
            definite C_1 and therefore signals numerical trouble upstream.
    """
    y_c_inv_y = quad - t_dot_x
    if y_c_inv_y <= 0.0:
        msg = (f'Marginal-likelihood quadratic form y^T C^-1 y = {y_c_inv_y:.6e} is not positive '
               f'(sum_i y_i^T Lambda_i^-1 y_i = {quad:.6e}, t^T Sigma_M^-1 t = {t_dot_x:.6e}). '
               'C is positive definite by construction, so this indicates loss of precision in the '
               'assembly or the solve -- try raising jit.')
        raise RuntimeError(msg)
    return y_c_inv_y / (lam * n_ao)


def jitter_scale(mat):
    """Reference magnitude a relative jitter is measured against: the mean diagonal entry.

    `o.jit` is a *relative* jitter. The matrices it regularizes here span many orders of magnitude, 
    therefore the absolute jitter must scale with the matrix. The mean diagonal entry is a simple
    and robust measure of the matrix's scale, and it is guaranteed to be positive for a 
    positive-definite matrix.

    Args:
        mat (np.ndarray): Symmetric matrix whose diagonal sets the scale.

    Returns:
        float: Mean diagonal entry, or 1.0 if that is not a usable positive scale (which would
        make the relative jitter meaningless, so it degrades to an absolute one).
    """
    scale = float(np.trace(mat)) / mat.shape[0]
    if not np.isfinite(scale) or scale <= 0.0:
        logger.error('jitter_scale: mean diagonal is not positive finite, using 1.0 instead')
        scale = 1.0
    return scale


def kmm_cholesky(basis, ref_elem, k_mm_tmap, jit):
    """Build the dense K_MM matrix and its lower Cholesky factor.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        k_mm_tmap (metatensor.TensorMap): Reference-reference kernel, as loaded from p.kernel_mm.
        jit (float): Relative diagonal jitter added for numerical stability
            before the Cholesky factorization.

    Returns:
        tuple[np.ndarray, np.ndarray]: Dense, jittered K_MM (nao_ref, nao_ref), and its lower
        Cholesky factor.
    """
    k_mm_dense = kernel_block_to_dense_self(basis, ref_elem, k_mm_tmap)
    k_mm_dense[np.diag_indices_from(k_mm_dense)] += jit * jitter_scale(k_mm_dense)
    l_mm = spl.cholesky(k_mm_dense, lower=True)
    return k_mm_dense, l_mm


def robust_cholesky(mat, jit, max_tries=10):
    """Cholesky-factorize a symmetric matrix, growing a relative diagonal jitter until it succeeds.

    Args:
        mat (np.ndarray): Symmetric matrix (only the lower triangle contributes; the whole array
            must be finite), restored on exit.
        jit (float): Relative diagonal jitter added for numerical stability
        max_tries (int): Maximum number of ×10 escalations of the jitter before giving up.

    Returns:
        tuple[np.ndarray, float]: Lower Cholesky factor, and the *relative* jitter that succeeded.

    Raises:
        scipy.linalg.LinAlgError: Still indefinite after max_tries escalations.
    """
    diag = np.diag_indices_from(mat)
    scale = jitter_scale(mat)
    saved = mat[diag].copy()
    cur_jit = jit
    try:
        for i in range(max_tries):
            mat[diag] = saved + cur_jit * scale
            try:
                return spl.cholesky(mat, lower=True), cur_jit
            except spl.LinAlgError:
                if i == max_tries - 1:
                    raise
                # A zero starting jitter has nothing to escalate from: step to the roundoff floor.
                cur_jit = cur_jit * 10 if cur_jit > 0.0 else np.finfo(float).eps
    finally:
        mat[diag] = saved


def molecule_lambda_chol_knm(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, eta, jit):
    """Compute the Cholesky factor of Lambda_i and K_{I_i,M} for one training molecule.

    Handing out L_i lets the callers accumulate (L_i^-1 K_{I_i,M})^T (L_i^-1 K_{I_i,M}), an outer
    product that is *exactly* symmetric and PSD in floating point, contrary to the direct inverse approach

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        mol_idx (int): Dataset index of the training molecule.
        atoms_i (np.ndarray[int]): Atomic numbers of the training molecule.
        paths (SimpleNamespace): Configured paths and path templates.
        l_mm (np.ndarray): Lower Cholesky factor of the dense K_MM, from kmm_cholesky().
        eta (float): Noise scale (the theory document's eta; callers pass o.reg -- eta and the
            SA-GPR path's regularization coefficient are the same symbol in the theory, see
            regression.py's PITC branch).
        jit (float): Relative diagonal jitter added for numerical stability

    Returns:
        tuple[np.ndarray, np.ndarray]: lower Cholesky factor of Lambda_i (nao_i, nao_i) and
        K_{I_i,M} (nao_i, nao_ref).
    """
    mol_i = make_dummy_mol(atoms_i, basis=basis.basisname, ignore=True)
    metric_i = tensormap_to_array(mol_i, metatensor.load(paths.metric_matrix.format(mol_idx)), dest='gpr', fast=True)
    metric_i[np.diag_indices_from(metric_i)] += jit * jitter_scale(metric_i)

    k_nm_i = kernel_block_to_dense_rect(basis, atoms_i, ref_elem, metatensor.load(paths.kernel_nm.format(mol_idx)))

    idx_i = [(q, mol_idx, iat) for iat, q in enumerate(atoms_i)]
    # merge_ref_ps builds a block for every (l,q) in the passed lmax, so it must be scoped to this
    # molecule's own elements -- passing the full basis.lmax would KeyError on any element the
    # molecule doesn't contain (e.g. no oxygen).
    lmax_i = {q: basis.lmax[q] for q in np.unique(atoms_i)}
    power_i = merge_ref_ps(lmax_i, idx_i, paths.power_spectrum)
    kself_i = kernel_block_to_dense_self(basis, atoms_i, kernel_mm(lmax_i, power_i))

    # L_MM^-1 K_{M,I_i}
    u_i = spl.solve_triangular(l_mm, k_nm_i.T, lower=True)

    # d_i = K_i - K_{M,I_i}^T K_MM^-1 K_{M,I_i} = K_i - (L_MM^-1 K_{M,I_i})^T L_MM^-1 K_{M,I_i}. Exactly symmetric
    d_i = kself_i - u_i.T @ u_i
    metric_inv_i = spl.cho_solve(spl.cho_factor(metric_i), np.eye(metric_i.shape[0]))
    # 0.5*(metric_inv_i + metric_inv_i.T) ensure the symmetry of lambda. 
    lambda_i = d_i + eta * 0.5*(metric_inv_i + metric_inv_i.T)
    l_lambda_i, _ = robust_cholesky(lambda_i, jit)

    return l_lambda_i, k_nm_i
