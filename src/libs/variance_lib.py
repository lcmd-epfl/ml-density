"""Compute PITC whole-molecule predictive variance for query molecules.

Only scalar whole-molecule numbers are computed here. The spatially resolved field
sigma[rho(r)] = sqrt(phi(r)^T Sigma_c* phi(r)) is a separate, on-demand computation reserved
for extract_cube.py --std.

The quantity of interest is the predictive variance relative to a metric: V(A*) = Tr(Sigma_c* @ metric)
"""

import os
import logging
import numpy as np
import scipy.linalg as spl
import metatensor
from qstack import reorder
from libs.kernels_lib import kernel_block_to_dense_rect, kernel_block_to_dense_self, kernel_mm
from libs.tmap import merge_ref_ps
from libs.functions import make_pyscf_mol
from libs.config_utils import GPR_DTC

logger = logging.getLogger('__main__')


def load_sigma_f2(path):
    """Load the kernel amplitude fitted by regression.py.

    Args:
        path (str): Formatted path to the amplitude file (p.sigma_f2).

    Returns:
        float: The fitted sigma_f^2, or 1.0 when the file is absent -- which is the uncalibrated
        amplitude implied by the normalized kernel, i.e. the behaviour of runs predating the fit.
    """
    if not os.path.exists(path):
        logger.error(f'{path} does not exist -- falling back to the unfitted sigma_f^2 = 1, the amplitude '
                       'implied by the normalized kernel. Re-run regression.py to fit it.')
        return 1.0
    return float(np.loadtxt(path))


def variance_scale(o):
    """Scaling of the K_*M (L L^T)^-1 K_M* term implied by which matrix regression.py factorized.

    Both GP models evaluate Sigma_c* = sigma_p^2 [K_** - Q_** + K_*M Sigma_M^-1 K_M*], but
    they persist Cholesky factors of different matrices. gpr_PITC factorizes Sigma_M itself, so the term
    comes out directly. gpr_DTC factorizes the SA-GPR matrix A = eta*Sigma_M, so its V^T V is
    K_*M A^-1 K_M* = K_*M Sigma_M^-1 K_M* / eta and has to be multiplied back by eta.

    Args:
        o (SimpleNamespace): Configured options (reads o.regression_model, o.reg).

    Returns:
        float: The multiplier to pass to compute_molecule_sigma as var_scale.
    """
    return o.reg if o.regression_model==GPR_DTC else 1.0


def compute_molecule_sigma(basis, atoms_star, mol_idx, ref_elem, l_factor, l_mm, path_kern, path_ps, sigma_f2=1.0, var_scale=1.0):
    """Reconstruct one query molecule's dense predictive AO covariance Sigma_c*.

    Sigma_c* = sigma_f^2 [(K_{N*,N*} - Q_{N*,N*}) + var_scale * K_{N*,M} (L L^T)^-1 K_{N*,M}^T],
    with L the persisted Cholesky factor from regression.py and
    Q_{N*,N*} = K_{N*,M} K_MM^-1 K_{N*,M}^T the inducing-point approximation of the prior.

    The same expression serves DTC and PITC : the two differ only in the
    matrix L factorizes, which var_scale compensates -- see variance_scale(). The leading
    K_** - Q_** residual is what both have and SA-GPR does not, and is why their predictive variance
    grows back toward the prior far from the reference set instead of collapsing.

    The kernel amplitude enters as a single multiplicative factor (see pitc_lib.fit_sigma_f2), so it
    is applied here, once, rather than in each caller -- that keeps the scalar trace computed by
    variance.py and the spatial std field computed by extract_cube.py --std consistently calibrated.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        atoms_star (np.ndarray[int]): Atomic numbers of the query molecule.
        mol_idx (int): Dataset index of the query molecule.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        l_factor (np.ndarray): Persisted lower Cholesky factor from regression.py (nao_ref, nao_ref):
            of Sigma_M for gpr_PITC, of the SA-GPR matrix eta*Sigma_M for gpr_DTC.
        l_mm (np.ndarray): Lower Cholesky factor of the jittered K_MM, from pitc_lib.kmm_cholesky.
        path_kern (str): Template path to K_{query,M} kernel files.
        path_ps (str): Template path to the query molecule's own power spectrum.
        sigma_f2 (float): Fitted kernel amplitude, from load_sigma_f2(). Defaults to 1.0, the
            uncalibrated prior amplitude implied by the normalized kernel.
        var_scale (float): Multiplier undoing l_factor's scaling, from variance_scale().

    Returns:
        np.ndarray: Dense (nao_star, nao_star) predictive covariance matrix, GPR AO order.
    """
    kmat_star = kernel_block_to_dense_rect(basis, atoms_star, ref_elem, metatensor.load(path_kern.format(mol_idx)))

    lmax_star = {q: basis.lmax[q] for q in np.unique(atoms_star)}
    idx_star = [(q, mol_idx, iat) for iat, q in enumerate(atoms_star)]
    power_star = merge_ref_ps(lmax_star, idx_star, path_ps)
    kself_star = kernel_block_to_dense_self(basis, atoms_star, kernel_mm(lmax_star, power_star))

    u = spl.solve_triangular(l_mm, kmat_star.T, lower=True)      # U^T U = Q_{N*,N*}
    v = spl.solve_triangular(l_factor, kmat_star.T, lower=True)  # var_scale * V^T V = K_*M Sigma_M^-1 K_M*
    return sigma_f2 * (kself_star - u.T @ u + var_scale * (v.T @ v))


def molecule_variance_trace(sigma_star, metric_star):
    """Contract the predictive AO covariance against an AO metric: Tr(Sigma_c* M_A*).

    Both arguments must be in the same AO order -- the trace is invariant under a simultaneous
    permutation of the two, but not under permuting only one.

    Args:
        sigma_star (np.ndarray): Dense (nao_star, nao_star) predictive AO covariance, GPR AO
            order, as returned by compute_molecule_sigma.
        metric_star (np.ndarray): Dense (nao_star, nao_star) AO metric matrix M_A*, GPR AO order.

    Returns:
        float: The contracted scalar, in the units implied by metric_star.
    """
    return float(np.sum(sigma_star * metric_star.T))


def density_self_repulsion(coeffs, metric):
    """Self-repulsion of a fitted density:  c^T J c.

    Args:
        coeffs (np.ndarray): Density-fitting coefficients, GPR AO order.
        metric (np.ndarray): Dense (nao, nao) 2-center Coulomb metric, GPR AO order.

    Returns:
        float: The self-repulsion, in Hartree -- the same units as Tr(Sigma_c* J), so the ratio
        of the two is dimensionless.
    """
    return float(coeffs @ metric @ coeffs)


def compute_coulomb_metric(atoms_star, positions_star, basisname):
    """Compute a query molecule's 2-center Coulomb metric directly via PySCF.

    J depends only on geometry and basis, so it can be evaluated for any molecule -- train, test
    or extrapolated -- without a reference DFT calculation. It is computed rather than loaded from
    p.metric_matrix so that extrapolation molecules, which have no stored metric, take the same
    code path; the stored jmat_* inputs are the same integral to ~1e-13.

    Args:
        atoms_star (np.ndarray[int]): Atomic numbers of the query molecule.
        positions_star (np.ndarray): Atomic coordinates, shape (natoms, 3).
        basisname (str): Basis-set name.

    Returns:
        np.ndarray: The (nao_star, nao_star) Coulomb metric
        J_ij = int int phi_i(r) phi_j(r')/|r-r'| dr dr', in GPR AO order.
    """
    mol = make_pyscf_mol(atoms_star, positions_star, basis=basisname, ignore=True)
    return reorder.reorder_ao(mol, mol.intor('int2c2e_sph'), src='pyscf', dest='gpr')
