"""Compute PITC whole-molecule predictive variance for query molecules.

Only the scalar whole-molecule variance Tr(Sigma_c* S_A*) is computed here. The spatially
resolved variance field Var[rho(r)] = phi(r)^T Sigma_c* phi(r) is a separate, on-demand
computation reserved for extract_cube.py --variance.
"""

import numpy as np
import scipy.linalg as spl
import metatensor
from qstack import reorder
from qstack.io.metatensor import tensormap_to_array
from libs.kernels_lib import kernel_block_to_dense_rect, kernel_block_to_dense_self, kernel_mm
from libs.tmap import merge_ref_ps
from libs.functions import make_dummy_mol, make_pyscf_mol


def compute_molecule_sigma(basis, atoms_star, mol_idx, ref_elem, l_factor, path_kern, path_ps):
    """Reconstruct one query molecule's dense PITC predictive AO covariance Sigma_c*.

    Sigma_c* = K_{N*,N*} - K_{N*,M} (L L^T)^-1 K_{N*,M}^T, with L the persisted PITC Cholesky
    factor of (K_NM^T Lambda^-1 K_NM + K_MM).

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        atoms_star (np.ndarray[int]): Atomic numbers of the query molecule.
        mol_idx (int): Dataset index of the query molecule.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        l_factor (np.ndarray): Persisted lower PITC Cholesky factor (totsize, totsize).
        path_kern (str): Template path to K_{query,M} kernel files.
        path_ps (str): Template path to the query molecule's own power spectrum.

    Returns:
        np.ndarray: Dense (nao_star, nao_star) predictive covariance matrix, GPR AO order.
    """
    kmat_star = kernel_block_to_dense_rect(basis, atoms_star, ref_elem, metatensor.load(path_kern.format(mol_idx)))

    lmax_star = {q: basis.lmax[q] for q in np.unique(atoms_star)}
    idx_star = [(q, mol_idx, iat) for iat, q in enumerate(atoms_star)]
    power_star = merge_ref_ps(lmax_star, idx_star, path_ps)
    kself_star = kernel_block_to_dense_self(basis, atoms_star, kernel_mm(lmax_star, power_star))

    v = spl.solve_triangular(l_factor, kmat_star.T, lower=True)
    return kself_star - v.T @ v


def molecule_variance_trace(sigma_star, s_star):
    """Whole-molecule variance V(A*) = Tr(Sigma_c* S_A*) = integral over space of Var[rho(r)].

    Args:
        sigma_star (np.ndarray): Dense (nao_star, nao_star) predictive AO covariance, GPR AO
            order, as returned by compute_molecule_sigma.
        s_star (np.ndarray): Dense (nao_star, nao_star) AO metric matrix S_A*, GPR AO order.

    Returns:
        float: Whole-molecule predictive variance.
    """
    return float(np.sum(sigma_star * s_star.T))


def load_reference_metric(atoms_star, mol_idx, basisname, path_metric):
    """Load a train/test query molecule's AO metric matrix S_A* from its preprocessed file.

    Args:
        atoms_star (np.ndarray[int]): Atomic numbers of the query molecule.
        mol_idx (int): Dataset index of the query molecule.
        basisname (str): Basis-set name.
        path_metric (str): Template path to p.metric_matrix.

    Returns:
        np.ndarray: Dense (nao_star, nao_star) AO metric matrix, GPR AO order.
    """
    mol_star = make_dummy_mol(atoms_star, basis=basisname, ignore=True)
    return tensormap_to_array(mol_star, metatensor.load(path_metric.format(mol_idx)), dest='gpr', fast=True)


def compute_reference_metric(atoms_star, positions_star, basisname):
    """Compute an extrapolated query molecule's AO metric matrix S_A* directly via PySCF.

    Extrapolated/out-of-sample molecules have no externally supplied metric-matrix file, unlike
    train/test molecules (see preprocess.py, which reads it from p.input_metrics). Verified
    against a stored p.metric_matrix file: S_A* is the 2-center Coulomb (RI) metric
    (n,l,m)_i,(n',l',m')_j -> int phi_i(r) phi_j(r') / |r-r'| dr dr' -- PySCF's 'int2c2e_sph' --
    not the plain overlap ('int1e_ovlp_sph'), which differs from the stored files by ~100%. Both
    only depend on geometry and basis, so this can be evaluated for any molecule without needing
    a reference DFT calculation.

    Args:
        atoms_star (np.ndarray[int]): Atomic numbers of the query molecule.
        positions_star (np.ndarray): Atomic coordinates, shape (natoms, 3).
        basisname (str): Basis-set name.

    Returns:
        np.ndarray: Dense (nao_star, nao_star) AO metric matrix, GPR AO order.
    """
    mol = make_pyscf_mol(atoms_star, positions_star, basis=basisname, ignore=True)
    s_star = mol.intor('int2c2e_sph')
    return reorder.reorder_ao(mol, s_star, src='pyscf', dest='gpr')
