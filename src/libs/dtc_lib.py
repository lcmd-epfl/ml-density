"""DTC (Deterministic Training Conditional) sparse GP -- the model of main.pdf Sec. IIE.

DTC applies the Nystrom replacement K_AB -> Q_AB = K_AR K_RR^-1 K_RB (Eq. 20) to the training
conditional only, leaving the test-test block K_** exact. Writing

    A = K_TR^T M_T K_TR + lambda K_RR,          lambda = sigma_rho^2 / sigma_p^2   (Eq. 17)

for the system matrix, the model is, entirely on its own terms:

    mu    = c_av + K_*R A^-1 K_TR^T M_T (c~_T - c_av)                              (Eq. 21)
    Sigma = sigma_p^2 [K_** - Q_** + lambda K_*R A^-1 K_*R^T]                      (Eq. 22)
    sigma_p^2 = (1/(lambda N_T)) [ (c~_T - c_av)^T M_T (c~_T - c_av) - ||y||^2 ]   (Eq. 26)

with y = L^-1 K_TR^T M_T (c~_T - c_av) and L the Cholesky factor of A.

Two consequences of this form drive the whole implementation:

- A involves the metric M_T directly, never its inverse. The observation noise of Eq. 15 is
  sigma_rho^2 M_T^-1, so the precision it contributes is M_T/sigma_rho^2 and the 1/lambda that
  carries is folded once into the lambda K_RR term instead of being divided out per molecule. That
  is why gram_matrix.do_work_gram contracts K^T M_i K straight from the stored metric, factorizing
  nothing.
- Eq. 21 does not contain sigma_p^2. The predictive mean is invariant to the prior scale, which is
  what lets Eq. 26 be evaluated after the solve, from quantities the solve already produced.

This module is deliberately free of any reference to Sigma_M, Lambda_i or D_i: DTC is not derived
here as a special case of anything. pitc_lib.py holds the separate PITC model, which is a stricter
approximation kept in the code but absent from main.pdf.
"""

import logging
import numpy as np
import scipy.linalg as spl
import scipy.optimize as spo
from libs.gp_common import robust_cholesky

logger = logging.getLogger('__main__')


REG_BOUNDS = (1e-8, 1.0)
REG_NGRID = 9
REG_XATOL = 0.05  # in log10(lambda), i.e. ~12% on lambda itself


def fit_prior_scale(quad, y_norm2, n_ao, lam):
    """Type-II maximum-likelihood estimate of the prior variance sigma_p^2 (main.pdf Eq. 26).

    Eq. 25 maximizes the marginal likelihood of Eq. 23 over sigma_p^2 in closed form, and the
    Woodbury identity turns the N_T x N_T inverse it contains into quantities the training solve has
    already produced:

        sigma_p^2 = (1/(lambda N_T)) [ (c~_T - c_av)^T M_T (c~_T - c_av) - ||y||^2 ]

    The first term is the squared metric norm of the reference densities measured from the
    prior-mean density, which preprocess.py tabulates per molecule in p.coef_norms[:, 1]. The second
    is ||L^-1 t||^2 = t^T A^-1 t, i.e. the target vector dotted with the solved weights. So the fit
    costs one sum and one inner product on top of the solve.

    Args:
        quad (float): (c~_T - c_av)^T M_T (c~_T - c_av), summed over the training molecules.
        y_norm2 (float): ||y||^2 = t^T A^-1 t, the target vector dotted with the solved weights.
        n_ao (float): N_T, the total number of training AO coefficients.
        lam (float): lambda, the regularization parameter of Eq. 17.

    Returns:
        float: The fitted sigma_p^2.

    Raises:
        RuntimeError: The bracket came out non-positive. It is a quadratic form of the positive
            definite Q_TT + lambda M_T^-1 (Eq. 25), so this signals loss of precision in the
            assembly or the solve -- try raising jit.
    """
    bracket = quad - y_norm2
    if bracket <= 0.0:
        msg = (f'Marginal-likelihood quadratic form came out non-positive ({bracket:.6e}): '
               f'(c~-c_av)^T M (c~-c_av) = {quad:.6e}, ||y||^2 = {y_norm2:.6e}. Eq. 25 is a '
               'quadratic form of a positive definite matrix, so this indicates loss of precision '
               'in the assembly or the solve -- try raising jit.')
        raise RuntimeError(msg)
    return bracket / (lam * n_ao)


def predictive_covariance(kself, u, v, sigma_p2, lam):
    """DTC predictive covariance of one query molecule's coefficients (main.pdf Eq. 22).

        Sigma = sigma_p^2 [K_** - Q_** + lambda K_*R A^-1 K_*R^T]

    with Q_** = U^T U and U = L_RR^-1 K_R*, and K_*R A^-1 K_*R^T = V^T V with V = L^-1 K_R*.

    The K_** - Q_** residual is what survives from leaving the test-test block exact, and is why the
    error bar grows back toward the prior sigma_p^2 K_** for a test environment the reference set
    spans poorly, instead of collapsing there as SoR's does.

    Args:
        kself (np.ndarray): Dense (nao_star, nao_star) exact self-kernel K_**.
        u (np.ndarray): L_RR^-1 K_R*, shape (nao_ref, nao_star).
        v (np.ndarray): L^-1 K_R* with L the Cholesky factor of A, shape (nao_ref, nao_star).
        sigma_p2 (float): The prior variance, fixed or from fit_prior_scale().
        lam (float): lambda, the regularization parameter of Eq. 17.

    Returns:
        np.ndarray: Dense (nao_star, nao_star) predictive covariance, in the AO order of kself.
    """
    return sigma_p2 * (kself - u.T @ u + lam * (v.T @ v))


def _add_scaled(mat, other, alpha, chunk=1024):
    """In-place `mat += alpha*other`, row block by row block.

    `mat += alpha*other` would materialize a second full matrix, which at M=512 is 8 GB. The blocked
    form keeps the temporary to `chunk` rows.

    Args:
        mat (np.ndarray): Updated in place.
        other (np.ndarray): Same shape as mat.
        alpha (float): Scaling of `other`.
        chunk (int): Number of rows per block.
    """
    for i in range(0, mat.shape[0], chunk):
        mat[i:i+chunk] += alpha * other[i:i+chunk]


def fit_regularisation(mat, k_mm, target, quad, n_ao, bounds=REG_BOUNDS, ngrid=REG_NGRID):
    """Choose lambda by maximizing the DTC marginal likelihood of main.pdf Eq. 23.

    Eq. 26 identifies only the *product* sigma_p^2 * lambda, which by Eq. 17 is sigma_rho^2, the
    noise variance. Measured on QM7, sigma_p^2 * lambda is constant to five significant figures over
    eight decades of lambda, so choosing lambda by hand chooses the reported error bar by hand.
    This fit removes that freedom.

    With sigma_p^2 profiled out at its Eq. 25 optimum and C_1 = Q_TT + lambda M_T^-1,

        -2 log p = N_T log sigma_p^2(lambda) + log|C_1(lambda)| + const,
        log|C_1| = (N_T - P) log lambda + log|A(lambda)| - log|K_RR| - sum_i log|M_i|

    the last two terms being independent of lambda and therefore dropped: only the optimum is
    wanted, not the value. What remains costs one Cholesky of A(lambda) per trial value, which is
    cheap because DTC's Gram matrix and target vector do not depend on lambda at all -- nothing is
    re-assembled.

    A coarse log grid is scanned first and the bracketing interval refined by a bounded Brent
    search, so a flat or multi-modal profile cannot strand the search in a local minimum.

    Note: main.pdf Sec. IIF prescribes lambda "by a one-dimensional scan or cross-validation of the
    predictive mean" instead. Eq. 23 is the paper's own marginal likelihood, and it is the only
    route with any grip on QM7, where the predictive mean is flat to three digits across the whole
    of `bounds` (measured: baselined MAE 48.9% at every point, M=128) so a CV scan has nothing to
    minimize. On small training sets the two agree in direction: on sidechains the fit picks
    lambda = 1.9e-3, where the baselined MAE is 9.2% against 131% at lambda = 1e-6.

    Args:
        mat (np.ndarray): On entry the (P, P) lower triangle of the Gram matrix with the jitter
            already on its diagonal, i.e. A(lambda=0). **Overwritten**: on exit it holds A at the
            last lambda evaluated, which is not necessarily the one returned, so the caller must
            rebuild the system matrix at the returned lambda before factorizing it for good.
        k_mm (np.ndarray): Dense (P, P) lower triangle of the unjittered K_RR.
        target (np.ndarray): Dense (P,) target vector K_TR^T M_T (c~_T - c_av).
        quad (float): (c~_T - c_av)^T M_T (c~_T - c_av), from p.ml_terms.
        n_ao (float): N_T, the total number of training AO coefficients, from p.ml_terms.
        bounds (tuple[float, float]): Search interval for lambda.
        ngrid (int): Number of coarse grid points spanning `bounds`.

    Returns:
        float: The fitted lambda.

    Raises:
        RuntimeError: Every grid point failed, so there is no usable profile to refine.
    """
    n_ao = float(n_ao)
    nao_ref = mat.shape[0]
    current = [0.0]  # the lambda currently inside `mat`

    def neg2_log_evidence(log10_lam):
        """Profile -2 log p at one lambda, up to the lambda-independent constants.

        Args:
            log10_lam (float): log10 of the trial lambda.

        Returns:
            float: The score to minimize, or +inf where the model is not evaluable.
        """
        lam = 10.0**float(log10_lam)
        _add_scaled(mat, k_mm, lam - current[0])
        current[0] = lam
        try:
            l_factor, _ = robust_cholesky(mat, 0.0)
        except spl.LinAlgError:
            logger.warning(f'  lambda = {lam:.4e}: A is not positive definite, skipped')
            return np.inf
        x = spl.cho_solve((l_factor, True), target)
        try:
            sigma_p2 = fit_prior_scale(quad, target @ x, n_ao, lam)
        except RuntimeError as exc:
            logger.warning(f'  lambda = {lam:.4e}: {exc}')
            return np.inf
        score = (n_ao*np.log(sigma_p2) + (n_ao - nao_ref)*np.log(lam)
                 + 2.0*np.sum(np.log(np.diag(l_factor))))
        logger.info(f'  lambda = {lam:.4e}   sigma_p^2 = {sigma_p2:.4e}   -2 log evidence = {score:.8e}')
        return score

    grid = np.linspace(np.log10(bounds[0]), np.log10(bounds[1]), ngrid)
    logger.info(f'fitting lambda by marginal likelihood over {bounds[0]:.0e} .. {bounds[1]:.0e} '
                f'({ngrid} grid points, then a bounded refinement)')
    scores = np.array([neg2_log_evidence(point) for point in grid])
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        msg = ('The DTC marginal likelihood could not be evaluated at any lambda in '
               f'{bounds[0]:.0e} .. {bounds[1]:.0e}. Check the Gram matrix and p.ml_terms, or set '
               'a fixed `regularisation` in the config.')
        raise RuntimeError(msg)
    if best in {0, ngrid-1}:
        logger.warning(f'the marginal likelihood optimum sits on the edge of the search interval '
                       f'({10.0**grid[best]:.2e}); the true optimum may lie outside '
                       f'{bounds[0]:.0e} .. {bounds[1]:.0e}')

    result = spo.minimize_scalar(neg2_log_evidence, method='bounded',
                                 bounds=(grid[max(best-1, 0)], grid[min(best+1, ngrid-1)]),
                                 options={'xatol': REG_XATOL})
    # The refinement is only trusted when it actually improved on the grid: `bounded` reports
    # success even when it terminates on the tolerance without beating its bracket.
    return float(10.0**(result.x if result.success and result.fun <= scores[best] else grid[best]))
