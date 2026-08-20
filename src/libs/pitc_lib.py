"""PITC (Partially Independent Training Conditional) sparse GP.

**Not part of main.pdf.** The paper's model is DTC (dtc_lib.py); PITC is kept here as the stricter
approximation of the same family, useful as a reference point and as the check that DTC's
assembly is right (tests/test_dtc_scaling.py). It is selected by regression_model = gpr_PITC.

Where DTC replaces the whole training conditional by its Nystrom projection, PITC keeps the
per-molecule residual, so the observation precision of training molecule i becomes

    D_i = K_{I_i,I_i} - K_{I_i,R} K_RR^-1 K_{R,I_i},        Lambda_i = D_i + eta*M_i^-1

and the model solves Sigma_M = K_RR + sum_i K_{R,I_i} Lambda_i^-1 K_{I_i,R} against
t = sum_i K_{R,I_i} Lambda_i^-1 y_i. Keeping D_i costs a metric inverse and a Lambda_i Cholesky per
molecule, neither of which DTC needs; setting D_i = 0 recovers DTC's system scaled by eta, which is
what test_dtc_scaling.py exploits, but DTC is implemented from its own equations and does not go
through this module.
"""

import logging
import numpy as np
import scipy.linalg as spl
import metatensor
from qstack.io.metatensor import tensormap_to_array
from libs.gp_common import jitter_scale, robust_cholesky
from libs.kernels_lib import kernel_block_to_dense_rect, kernel_block_to_dense_self, kernel_mm
from libs.tmap import merge_ref_ps
from libs.functions import make_dummy_mol

logger = logging.getLogger('__main__')


def fit_prior_scale(quad, y_norm2, n_ao):
    """Type-II maximum-likelihood estimate of the prior variance sigma_p^2, for PITC.

    Same closed form as dtc_lib.fit_prior_scale, in PITC's own scaling: the quadratic form is
    y^T C_1^-1 y = sum_i y_i^T Lambda_i^-1 y_i - t^T Sigma_M^-1 t, and because this branch solves
    Sigma_M itself rather than a multiple of it, there is no lambda to divide out.

    Args:
        quad (float): sum_i y_i^T Lambda_i^-1 y_i, accumulated by target_vector.do_work_target_pitc.
        y_norm2 (float): t^T Sigma_M^-1 t, the target vector dotted with the solved weights.
        n_ao (float): N_T, the total number of training AO coefficients.

    Returns:
        float: The fitted sigma_p^2.

    Raises:
        RuntimeError: The quadratic form came out non-positive, which is impossible for a positive
            definite C_1 and therefore signals numerical trouble upstream.
    """
    y_c_inv_y = quad - y_norm2
    if y_c_inv_y <= 0.0:
        msg = (f'Marginal-likelihood quadratic form y^T C^-1 y = {y_c_inv_y:.6e} is not positive '
               f'(sum_i y_i^T Lambda_i^-1 y_i = {quad:.6e}, t^T Sigma_M^-1 t = {y_norm2:.6e}). '
               'C is positive definite by construction, so this indicates loss of precision in the '
               'assembly or the solve -- try raising jit.')
        raise RuntimeError(msg)
    return y_c_inv_y / n_ao


def predictive_covariance(kself, u, v, sigma_p2):
    """PITC predictive covariance of one query molecule's coefficients.

        Sigma = sigma_p^2 [K_** - Q_** + K_*R Sigma_M^-1 K_R*]

    with Q_** = U^T U and U = L_RR^-1 K_R*, and K_*R Sigma_M^-1 K_R* = V^T V with V = L_Sigma^-1
    K_R*. No lambda appears: this branch factorizes Sigma_M directly, unlike DTC, whose Eq. 22
    carries an explicit lambda because it factorizes A.

    Args:
        kself (np.ndarray): Dense (nao_star, nao_star) exact self-kernel K_**.
        u (np.ndarray): L_RR^-1 K_R*, shape (nao_ref, nao_star).
        v (np.ndarray): L_Sigma^-1 K_R*, shape (nao_ref, nao_star).
        sigma_p2 (float): The prior variance, fixed or from fit_prior_scale().

    Returns:
        np.ndarray: Dense (nao_star, nao_star) predictive covariance, in the AO order of kself.
    """
    return sigma_p2 * (kself - u.T @ u + v.T @ v)


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
        l_mm (np.ndarray): Lower Cholesky factor of the dense K_MM, from gp_common.kmm_cholesky().
        eta (float): Noise scale. Callers pass o.reg, the same lambda of main.pdf Eq. 17 that
            DTC uses; here it scales the metric inverse inside Lambda_i rather than K_RR.
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
