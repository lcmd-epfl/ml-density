"""Compute the target vector (kernel-projected fit targets)."""

import logging
import numpy as np
import scipy.linalg as spl
import metatensor
from qstack.io.metatensor import tensormap_to_array
from libs.tmap import vector2tmap, tmap2vector

logger = logging.getLogger('__main__')


def print_batches(fracs, ntrains, path_template):
    """Log matrix-output batch boundaries and target files.

    The target vector/Gram matrix are computed via sum over training set molecules.
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


def do_work_target(conf, path_proj, path_kern, target_tmap):
    """Accumulate the target-vector contribution for one training molecule.

    Args:
        conf (int): Molecule configuration index.
        path_proj (str): Template path to projections.
        path_kern (str): Template path to kernels.
        target_tmap (metatensor.TensorMap): Accumulator TensorMap for the target vector.
    """
    proj = metatensor.load(path_proj.format(conf))
    k_NM = metatensor.load(path_kern.format(conf))
    for (l1, q1), pblock in proj.items():
        kblock = k_NM.block(o3_lambda=l1, center_type=q1)
        tblock = target_tmap.block(o3_lambda=l1, center_type=q1)
        tblock.values[...] += np.einsum('kmMr,kmn->rMn', kblock.values, pblock.values)


def do_work_target_pitc(mol_idx, paths, lambda_inv_i, k_nm_i, metric_i, mol_i, target_vec, ml_terms):
    """Accumulate the PITC target-vector and marginal-likelihood contributions for one training molecule.

    Computes t_i = K_{I_i,M}^T Lambda_i^-1 y_i, where y_i = metric_i^-1 w_i is molecule i's baselined
    coefficient vector and w_i its AO-basis projection (p.projection).

    Also accumulates the two terms needed for the closed-form kernel-amplitude estimate
    sigma_f^2 = (sum_i y_i^T Lambda_i^-1 y_i - t.x) / N, finished in regression.py once the solve
    x = Sigma_M^-1 t is available. Lambda_i^-1 y_i is already needed for the target vector, so the
    quadratic form costs one extra dot product per molecule.

    Args:
        mol_idx (int): Molecule index.
        paths (SimpleNamespace): Configured paths and path templates.
        lambda_inv_i (np.ndarray): Lambda_i^-1, from molecule_lambda_inv_knm().
        k_nm_i (np.ndarray): K_{I_i,M}, from molecule_lambda_inv_knm().
        metric_i (np.ndarray): Jittered metric matrix of molecule i, from molecule_lambda_inv_knm().
        mol_i (pyscf.gto.Mole): Dummy mol matching molecule i's AO layout, from molecule_lambda_inv_knm().
        target_vec (np.ndarray): Dense (nao_ref,) accumulator, updated in place.
        ml_terms (np.ndarray): Dense (2,) accumulator, updated in place: [sum_i y_i^T Lambda_i^-1 y_i,
            N = sum_i nao_i].
    """
    proj_i = tensormap_to_array(mol_i, metatensor.load(paths.projection.format(mol_idx)), dest='gpr', fast=True)
    metric_inv_w_i = spl.cho_solve(spl.cho_factor(metric_i), proj_i)  # y_i
    lambda_inv_y_i = lambda_inv_i @ metric_inv_w_i
    target_vec += k_nm_i.T @ lambda_inv_y_i
    ml_terms[0] += metric_inv_w_i @ lambda_inv_y_i
    ml_terms[1] += len(metric_inv_w_i)


def get_target_vector(basis, ref_elem, fracs, ntrains, training_idx, paths):
    """Build and save target vectors for all requested training fractions.

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
    print_batches(fracs, ntrains, paths.target_vec)

    nao_ref = basis.nao_for_mol(ref_elem)
    target_tmap = vector2tmap(ref_elem, basis.llist, np.zeros(nao_ref))

    for frac, ntrain in zip(fracs, ntrains, strict=True):
        for imol in range(ntrain[0], ntrain[1]):
            logger.info(f'{0:4d}: {imol:4d}', extra={'flush': True})
            do_work_target(training_idx[imol], paths.projection, paths.kernel_nm, target_tmap)
        target_vec = tmap2vector(ref_elem, basis.llist, target_tmap)
        np.savetxt(paths.target_vec.format(train_frac=frac), target_vec)
