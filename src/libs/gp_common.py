"""Numerics shared by both sparse-GP models, owned by neither.

Nothing here is specific to DTC (main.pdf Sec. IIE) or to PITC: these are the jitter convention and
the Cholesky factorizations that any of the models needs. They live apart so that dtc_lib.py and
pitc_lib.py never have to import from each other, which would suggest one is derived from the other.
"""

import logging
import numpy as np
import scipy.linalg as spl
from libs.kernels_lib import kernel_block_to_dense_self

logger = logging.getLogger('__main__')


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
