"""Compute the target vector (kernel-projected fit targets)."""

import logging
import numpy as np
import scipy.linalg as spl
import metatensor
from libs.functions import remove_averages
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


def do_work_target_pitc(mol_idx, paths, basis, atoms_i, av_coefs, l_lambda_i, v_i, target_vec, ml_terms):
    """Accumulate the PITC target-vector and marginal-likelihood contributions for one training molecule.

    Computes t_i = K_{I_i,M}^T Lambda_i^-1 y_i, where y_i = c_i - c_av is molecule i's baselined
    coefficient vector, rebuilt from p.clean_coefficients and p.spherical_averages exactly as
    preprocess.py built it before projecting.

    The SoR path contracts the kernel against the *projection* w_i = metric_i y_i (p.projection),
    so it never needs y_i itself; PITC does, because Lambda_i is not proportional to metric_i.
    Recovering it as metric_i^-1 w_i is algebraically equivalent but numerically lossy: the metric
    has cond ~1e7 here, and the round-trip returns y_i to only ~1e-6 relative accuracy (measured on
    QM7/cc-pvdz-jkfit) instead of the ~1e-16 the direct read gives. It also cost a Cholesky
    factorization and solve per molecule to recover a vector already on disk.

    Also accumulates the two terms needed for the closed-form kernel-amplitude estimate
    sigma_f^2 = (sum_i y_i^T Lambda_i^-1 y_i - t.x) / N, finished in regression.py once the solve
    x = Sigma_M^-1 t is available.

    Both are expressed through the half-solve z_i = L_i^-1 y_i, with L_i L_i^T = Lambda_i: the
    target contribution is v_i^T z_i and the quadratic form is z_i.z_i. Writing the latter as a sum
    of squares makes it non-negative in floating point by construction.

    Args:
        mol_idx (int): Molecule index.
        paths (SimpleNamespace): Configured paths and path templates.
        basis (.functions.Basis): Basis used for AO indexing.
        atoms_i (np.ndarray[int]): Atomic numbers of the training molecule.
        av_coefs (dict[int, np.ndarray]): Per-element average l=0 coefficients (p.spherical_averages).
        l_lambda_i (np.ndarray): Lower Cholesky factor of Lambda_i, from molecule_lambda_chol_knm().
        v_i (np.ndarray): L_i^-1 K_{I_i,M} (nao_i, nao_ref), shared with do_work_gram_pitc().
        target_vec (np.ndarray): Dense (nao_ref,) accumulator, updated in place.
        ml_terms (np.ndarray): Dense (2,) accumulator, updated in place: [sum_i y_i^T Lambda_i^-1 y_i,
            N = sum_i nao_i].
    """
    y_i = remove_averages(basis.index(atoms_i), np.load(paths.clean_coefficients.format(mol_idx)), av_coefs)
    z_i = spl.solve_triangular(l_lambda_i, y_i, lower=True)  # L_i^-1 y_i
    target_vec += v_i.T @ z_i
    ml_terms[0] += z_i @ z_i
    ml_terms[1] += len(y_i)


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
