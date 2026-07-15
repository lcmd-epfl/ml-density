"""Compute the "A vector" (kernel * projection)."""

import logging
import numpy as np
import scipy.linalg as spl
import metatensor
from qstack.io.metatensor import tensormap_to_array
from libs.tmap import vector2tmap, tmap2vector
from libs.functions import make_dummy_mol

logger = logging.getLogger('__main__')


def print_batches(fracs, ntrains, path_template):
    """Log matrix-output batch boundaries and target files.

    The Avec/Bmat are computed via sum over training set molecules.
    If there are two training set fraction used, e.g. 0.5 and 1.0,
    sum over the first 50% training molecules is computed first and saved (first batch).
    Then a sum over the last 50% training molecules is added (second batch).

    Args:
        fracs (np.ndarray[float]): Training set fractions.
        ntrains (list[tuple[int], tuple[int]]): Training set boundaries per batch.
        path_template (str): Template path for output file.
    """
    def _make_msg(i, frac, ntrain):
        return f'batch {i:2d} [{ntrain[0]}--{ntrain[1]}):\t {path_template.format(train_frac=frac)}'
    msg = '\n'.join(_make_msg(i, *batch) for i, batch in enumerate(zip(fracs, ntrains, strict=True)))
    logger.info(msg, extra={'flush': True})


def do_work_a(conf, path_proj, path_kern, Avec):
    """Accumulate the A vector contribution for one training molecule.

    Args:
        conf (int): Molecule configuration index.
        path_proj (str): Template path to projections.
        path_kern (str): Template path to kernels.
        Avec (metatensor.TensorMap): Accumulator TensorMap for A coefficients.
    """
    proj = metatensor.load(path_proj.format(conf))
    k_NM = metatensor.load(path_kern.format(conf))
    for (l1, q1), pblock in proj.items():
        kblock = k_NM.block(o3_lambda=l1, center_type=q1)
        ablock = Avec.block(o3_lambda=l1, center_type=q1)
        ablock.values[...] += np.einsum('kmMr,kmn->rMn', kblock.values, pblock.values)


def do_work_a_pitc(mol_idx, paths, lambda_inv_i, kmat_i, s_i, mol_i, bvec):
    """Accumulate the PITC b-vector contribution for one training molecule.

    Computes b_i = K_{I_i,M}^T Lambda_i^-1 S_i^-1 w_i, where w_i is molecule i's AO-basis
    projection (p.projection).

    Args:
        mol_idx (int): Molecule index.
        paths (SimpleNamespace): Configured paths and path templates.
        lambda_inv_i (np.ndarray): Lambda_i^-1, from molecule_lambda_inv_kmat().
        kmat_i (np.ndarray): K_{I_i,M}, from molecule_lambda_inv_kmat().
        s_i (np.ndarray): Jittered S_i, from molecule_lambda_inv_kmat().
        mol_i (pyscf.gto.Mole): Dummy mol matching molecule i's AO layout, from molecule_lambda_inv_kmat().
        bvec (np.ndarray): Dense (totsize,) accumulator, updated in place.
    """
    proj_i = tensormap_to_array(mol_i, metatensor.load(paths.projection.format(mol_idx)), dest='gpr', fast=True)
    sinv_w_i = spl.cho_solve(spl.cho_factor(s_i), proj_i)
    bvec += kmat_i.T @ (lambda_inv_i @ sinv_w_i)


def get_a(basis, ref_elem, fracs, ntrains, training_idx, paths):
    """Build and save A vectors for all requested training fractions.

    SoR only (full_GPR=False)

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        fracs (np.ndarray[float]): Training set fractions.
        ntrains (list[tuple[int], tuple[int]]): Training set boundaries
                per fraction batch corresponding to the new molecules wrt the previous batch.
        training_idx (np.ndarray[int]): Training molecules indices.
        paths (SimpleNamespace): Configured paths and path templates..
    """
    print_batches(fracs, ntrains, paths.avec)

    totsize = basis.nao_for_mol(ref_elem)
    mol = make_dummy_mol(ref_elem, basis=basis.basisname, ignore=True)
    A1 = vector2tmap(mol, np.zeros(totsize))

    for frac, ntrain in zip(fracs, ntrains, strict=True):
        for imol in range(ntrain[0], ntrain[1]):
            logger.info(f'{0:4d}: {imol:4d}', extra={'flush': True})
            do_work_a(training_idx[imol], paths.projection, paths.kernel_nm, A1)
        A = tmap2vector(mol, A1)
        np.savetxt(paths.avec.format(train_frac=frac), A)
