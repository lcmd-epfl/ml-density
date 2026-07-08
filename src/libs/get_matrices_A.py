"""Compute the "A vector" (kernel * projection)."""

import logging
import numpy as np
import metatensor
from libs.tmap import vector2tmap, tmap2vector

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


def do_work_a(conf, ref_elem, path_proj, path_kern, Avec):
    """Accumulate the A vector contribution for one training molecule.

    Args:
        conf (int): Molecule configuration index.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        path_proj (str): Template path to projections.
        path_kern (str): Template path to kernels.
        Avec (metatensor.TensorMap): Accumulator TensorMap for A coefficients.
    """
    proj = metatensor.load(path_proj.format(conf))
    k_NM = metatensor.load(path_kern.format(conf))
    for (l1, q1), pblock in proj.items():
        kblock = k_NM.block(o3_lambda=l1, center_type=q1)
        ablock = Avec.block(o3_lambda=l1, center_type=q1)
        for iiref1 in range(np.count_nonzero(ref_elem==q1)):
            dA = np.einsum('kmM,kmn->Mn', kblock.values[:,:,:,iiref1], pblock.values)
            ablock.values[iiref1,:,:] += dA


def get_a(basis, ref_elem, fracs, ntrains, training_idx, paths):
    """Build and save A vectors for all requested training fractions.

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
    A1 = vector2tmap(ref_elem, basis.llist, np.zeros(totsize))
    for frac, ntrain in zip(fracs, ntrains, strict=True):
        for imol in range(ntrain[0], ntrain[1]):
            logger.info(f'{0:4d}: {imol:4d}', extra={'flush': True})
            do_work_a(training_idx[imol], ref_elem, paths.projection, paths.kernel_nm, A1)
        A = tmap2vector(ref_elem, basis.llist, A1)
        np.savetxt(paths.avec.format(train_frac=frac), A)
