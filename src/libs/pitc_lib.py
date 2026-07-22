"""PITC (Partially Independent Training Conditional) sparsification helpers.

Shared by the full_gpr=True branches of target_vector.py/gram_matrix.py/regression.py to
build, per training molecule i, the precision matrix Lambda_i = D_i + eta*metric_i^-1, with
D_i = K_{I_i,I_i} - K_{I_i,M} K_MM^-1 K_{M,I_i} (all three matrices are block-diagonal per molecule),
and the shared K_MM Cholesky factor both target_vector.py and gram_matrix.py need.
"""

import numpy as np
import scipy.linalg as spl
import metatensor
from qstack.io.metatensor import tensormap_to_array
from libs.kernels_lib import kernel_block_to_dense_rect, kernel_block_to_dense_self, kernel_mm
from libs.tmap import merge_ref_ps
from libs.functions import make_dummy_mol


def kmm_cholesky(basis, ref_elem, k_mm_tmap, jit):
    """Build the dense K_MM matrix and its lower Cholesky factor.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        k_mm_tmap (metatensor.TensorMap): Reference-reference kernel, as loaded from p.kernel_mm.
        jit (float): Diagonal jitter added for numerical stability before the Cholesky factorization.

    Returns:
        tuple[np.ndarray, np.ndarray]: Dense, jittered K_MM (nao_ref, nao_ref), and its lower
        Cholesky factor.
    """
    k_mm_dense = kernel_block_to_dense_self(basis, ref_elem, k_mm_tmap)
    k_mm_dense[np.diag_indices_from(k_mm_dense)] += jit
    l_mm = spl.cholesky(k_mm_dense, lower=True)
    return k_mm_dense, l_mm


def robust_cholesky(mat, jit, max_tries=10):
    """Cholesky-factorize a symmetric matrix, growing a diagonal jitter until it succeeds.

    PITC's Lambda_i^-1-weighted accumulation can be extremely ill-conditioned (eigenvalues
    spanning many orders of magnitude across training molecules), so a single small fixed jitter
    is not robust across datasets/configs. Doubles the jitter each retry, starting from `jit`.

    Args:
        mat (np.ndarray): Symmetric matrix (only the lower triangle is read), not modified in place.
        jit (float): Initial diagonal jitter to try.
        max_tries (int): Maximum number of doublings before giving up.

    Returns:
        tuple[np.ndarray, float]: Lower Cholesky factor, and the jitter value that succeeded.
    """
    cur_jit = jit
    eye = np.eye(mat.shape[0])
    for _ in range(max_tries):
        try:
            return spl.cholesky(mat + cur_jit*eye, lower=True), cur_jit
        except spl.LinAlgError:
            cur_jit *= 10
    return spl.cholesky(mat + cur_jit*eye, lower=True), cur_jit


def molecule_lambda_inv_knm(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, eta, jit):
    """Compute Lambda_i^-1, K_{I_i,M} and metric_i for one training molecule.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        mol_idx (int): Dataset index of the training molecule.
        atoms_i (np.ndarray[int]): Atomic numbers of the training molecule.
        paths (SimpleNamespace): Configured paths and path templates.
        l_mm (np.ndarray): Lower Cholesky factor of the dense K_MM, from kmm_cholesky().
        eta (float): Noise scale (the theory document's eta; callers pass o.reg -- eta and the
            SoR path's regularization coefficient are the same symbol in the theory, see
            regression.py's PITC branch).
        jit (float): Diagonal jitter added to metric_i before inverting it. metric_i is often severely
            ill-conditioned for redundant density-fitting auxiliary bases (e.g. cc-pvqz-jkfit,
            condition numbers ~1e7 observed), so its raw inverse amplifies floating-point noise by
            a comparable factor; jittering caps this the same way kmm_cholesky() does for K_MM.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, pyscf.gto.Mole]: Lambda_i^-1 (nao_i, nao_i),
        K_{I_i,M} (nao_i, nao_ref), the jittered metric_i (nao_i, nao_i)
    """
    mol_i = make_dummy_mol(atoms_i, basis=basis.basisname, ignore=True)
    metric_i = tensormap_to_array(mol_i, metatensor.load(paths.metric_matrix.format(mol_idx)), dest='gpr', fast=True)
    metric_i[np.diag_indices_from(metric_i)] += jit

    k_nm_i = kernel_block_to_dense_rect(basis, atoms_i, ref_elem, metatensor.load(paths.kernel_nm.format(mol_idx)))

    idx_i = [(q, mol_idx, iat) for iat, q in enumerate(atoms_i)]
    # merge_ref_ps builds a block for every (l,q) in the passed lmax, so it must be scoped to this
    # molecule's own elements -- passing the full basis.lmax would KeyError on any element the
    # molecule doesn't contain (e.g. no oxygen).
    lmax_i = {q: basis.lmax[q] for q in np.unique(atoms_i)}
    power_i = merge_ref_ps(lmax_i, idx_i, paths.power_spectrum)
    kself_i = kernel_block_to_dense_self(basis, atoms_i, kernel_mm(lmax_i, power_i))

    y_i = spl.cho_solve((l_mm, True), k_nm_i.T)              # K_MM^-1 K_{M,I_i}
    d_i = kself_i - k_nm_i @ y_i

    metric_inv_i = spl.cho_solve(spl.cho_factor(metric_i), np.eye(metric_i.shape[0]))
    lambda_i = d_i + eta * metric_inv_i
    lambda_inv_i = np.linalg.inv(lambda_i)

    return lambda_inv_i, k_nm_i, metric_i, mol_i
