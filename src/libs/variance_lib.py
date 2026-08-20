"""Whole-molecule predictive variance for query molecules, for either sparse-GP model.

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
from libs import dtc_lib, pitc_lib
from libs.timing import StepTimer

logger = logging.getLogger('__main__')


def load_prior_scale(path):
    """Load the prior variance sigma_p^2 that regression.py fitted (main.pdf Eq. 26).

    Args:
        path (str): Formatted path to the prior-scale file (p.sigma_p2).

    Returns:
        float: The fitted sigma_p^2.

    Raises:
        RuntimeError: The file does not exist. There is deliberately no fallback: sigma_p^2 is the
            single overall factor on Eq. 22, so substituting a placeholder would not degrade the
            error bars, it would silently rescale every one of them.
    """
    if not os.path.exists(path):
        msg = (f'{path} does not exist -- sigma_p^2 comes from regression.py, which writes it when '
               '`prior_scale = fit`. Re-run it for this training fraction, or set a fixed '
               '`prior_scale` in the config.')
        raise RuntimeError(msg)
    return float(np.loadtxt(path))


def load_fitted_reg(path):
    """Load the lambda that regression.py chose by marginal likelihood (main.pdf Eq. 23).

    Args:
        path (str): Formatted path to the fitted-lambda file (p.fitted_reg).

    Returns:
        float: The fitted lambda.

    Raises:
        RuntimeError: The file does not exist. The configured `regularisation = fit` says the
            number is not known ahead of the fit, and guessing one would mis-scale the second term
            of Eq. 22 by that guess.
    """
    if not os.path.exists(path):
        msg = (f'{path} does not exist -- the config sets `regularisation = fit`, so lambda comes '
               'from regression.py. Re-run it for this training fraction.')
        raise RuntimeError(msg)
    return float(np.loadtxt(path))


def prior_scale(o, p, frac):
    """The prior variance sigma_p^2 to use, whether configured or fitted.

    Args:
        o (SimpleNamespace): Configured options (reads o.fit_sigma_p2, o.sigma_p2).
        p (SimpleNamespace): Configured paths (reads p.sigma_p2).
        frac (float): Training fraction whose model is being evaluated.

    Returns:
        float: sigma_p^2, the overall factor of main.pdf Eq. 22.
    """
    return load_prior_scale(p.sigma_p2.format(train_frac=frac)) if o.fit_sigma_p2 else o.sigma_p2


def dtc_lambda(o, p, frac):
    """The regularization parameter lambda to use, whether configured or fitted.

    lambda appears explicitly in DTC's predictive covariance (main.pdf Eq. 22), multiplying the
    K_*R A^-1 K_*R^T term, because A = K_TR^T M_T K_TR + lambda K_RR is what regression.py
    factorized. Under `regularisation = fit` it is whatever the marginal-likelihood fit settled on,
    per training fraction, so it is read back from disk rather than taken from o.reg (None then).

    Args:
        o (SimpleNamespace): Configured options (reads o.fit_reg, o.reg).
        p (SimpleNamespace): Configured paths (reads p.fitted_reg).
        frac (float): Training fraction whose model is being evaluated.

    Returns:
        float: lambda, as defined by main.pdf Eq. 17.
    """
    return load_fitted_reg(p.fitted_reg.format(train_frac=frac)) if o.fit_reg else o.reg


def query_kernel_blocks(basis, atoms_star, mol_idx, ref_elem, l_factor, l_mm, path_kern, path_ps, timer=None):
    """Build the three kernel quantities both models' predictive covariances are made of.

    This is the expensive part and it is model-independent: the exact self-kernel K_**, and the two
    triangular solves U = L_RR^-1 K_R* and V = L^-1 K_R* against the two persisted Cholesky factors.
    Which formula they are then combined by is the model's own business -- dtc_lib and pitc_lib each
    hold theirs.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        atoms_star (np.ndarray[int]): Atomic numbers of the query molecule.
        mol_idx (int): Dataset index of the query molecule.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        l_factor (np.ndarray): Persisted lower Cholesky factor from regression.py (nao_ref,
            nao_ref): of A for gpr_DTC, of Sigma_M for gpr_PITC.
        l_mm (np.ndarray): Lower Cholesky factor of the jittered K_RR, from gp_common.kmm_cholesky.
        path_kern (str): Template path to K_{query,R} kernel files.
        path_ps (str): Template path to the query molecule's own power spectrum.
        timer (.timing.StepTimer | None): Optional profiler; None disables the instrumentation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: K_** (nao_star, nao_star), U and V, both
        (nao_ref, nao_star).
    """
    timer = timer if timer is not None else StepTimer(enabled=False)

    with timer.step('sigma: load K_*M tensormap'):
        k_tmap = metatensor.load(path_kern.format(mol_idx))
    with timer.step('sigma: densify K_*M'):
        kmat_star = kernel_block_to_dense_rect(basis, atoms_star, ref_elem, k_tmap)

    lmax_star = {q: basis.lmax[q] for q in np.unique(atoms_star)}
    idx_star = [(q, mol_idx, iat) for iat, q in enumerate(atoms_star)]
    with timer.step('sigma: read own power spectrum (merge_ref_ps)'):
        power_star = merge_ref_ps(lmax_star, idx_star, path_ps)
    with timer.step('sigma: self-kernel K_** + densify'):
        kself_star = kernel_block_to_dense_self(basis, atoms_star, kernel_mm(lmax_star, power_star))

    with timer.step('sigma: triangular solve vs L_MM (Nystrom Q_**)'):
        u = spl.solve_triangular(l_mm, kmat_star.T, lower=True)      # U^T U = Q_**
    with timer.step('sigma: triangular solve vs L (posterior term)'):
        v = spl.solve_triangular(l_factor, kmat_star.T, lower=True)  # V^T V = K_*R (L L^T)^-1 K_R*
    return kself_star, u, v


def compute_molecule_sigma(model, basis, atoms_star, mol_idx, ref_elem, l_factor, l_mm, path_kern,
                           path_ps, sigma_p2, lam=None, timer=None):
    """Reconstruct one query molecule's dense predictive AO covariance, by the model's own equation.

    Builds the shared kernel blocks and hands them to gpr_DTC's main.pdf Eq. 22 or to gpr_PITC's
    counterpart. sigma_p^2 is applied here, once, rather than in each caller, so that the scalar
    trace computed by variance.py and the spatial std field computed by extract_cube.py --std stay
    consistently calibrated.

    Args:
        model (str): o.regression_model, one of GPR_DTC / GPR_PITC.
        basis (.functions.Basis): Basis used for AO indexing.
        atoms_star (np.ndarray[int]): Atomic numbers of the query molecule.
        mol_idx (int): Dataset index of the query molecule.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        l_factor (np.ndarray): Persisted lower Cholesky factor from regression.py.
        l_mm (np.ndarray): Lower Cholesky factor of the jittered K_RR.
        path_kern (str): Template path to K_{query,R} kernel files.
        path_ps (str): Template path to the query molecule's own power spectrum.
        sigma_p2 (float): The prior variance, from prior_scale().
        lam (float | None): lambda, from dtc_lambda(). Required for gpr_DTC, unused by gpr_PITC.
        timer (.timing.StepTimer | None): Optional profiler; None disables the instrumentation.

    Returns:
        np.ndarray: Dense (nao_star, nao_star) predictive covariance matrix, GPR AO order.

    Raises:
        ValueError: gpr_DTC was asked for without a lambda.
    """
    timer = timer if timer is not None else StepTimer(enabled=False)
    kself_star, u, v = query_kernel_blocks(basis, atoms_star, mol_idx, ref_elem, l_factor, l_mm,
                                           path_kern, path_ps, timer)
    with timer.step('sigma: assemble U^T U / V^T V'):
        if model != GPR_DTC:
            return pitc_lib.predictive_covariance(kself_star, u, v, sigma_p2)
        if lam is None:
            msg = 'gpr_DTC needs lambda: Eq. 22 carries it explicitly. Pass dtc_lambda(o, p, frac).'
            raise ValueError(msg)
        return dtc_lib.predictive_covariance(kself_star, u, v, sigma_p2, lam)


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
