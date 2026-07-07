"""Compute the "B matrix" (kernel * metric matrix * kernel)."""

import logging
import numpy as np
import metatensor
from libs.get_matrices_A import print_batches
from libs.multi import print_nodes, scatter_jobs

logger = logging.getLogger('__main__')


def print_mem(totsize, ntrain):
    """Log estimated memory usage to store the result.

    Args:
        totsize (int): Problem dimensionality (number of AO coefficients).
        ntrain (int): Number of training molecules.
    """
    b2mib = 1.0/(1<<20)
    b2gib = 1.0/(1<<30)
    size = symsize(totsize)*np.array(0.0).itemsize
    logger.info(f"""Problem dimensionality = {totsize}\n\
Number of training molecules = {ntrain}\n\
output: {size:16d} bytes ({size*b2mib:10.2f} MiB, {size*b2gib:6.2f} GiB)\n""", extra={'flush': True})


def symsize(M):
    """Compute the size of an 1D array representing a symmetric matrix.

    Args:
        M (int): Matrix dimension.

    Returns:
        int: Size M*(M+1)/2.
    """
    return (M*(M+1))//2


def mpos(i, j):
    """Map a symmetric matrix index to an index in a corresponding 1D array.

    A symmetric matrix is represented as a flattened lower-triangular matrix.

    Args:
        i (int): i index, i<=j.
        j (int): j index, 0<=j<M (matrix size).

    Returns:
        int: Index corresponding to [i,j] and [j,i] value.

    Note:
        There is no check that `i<=j`.
    """
    return i + ((j*(j+1))//2)


def do_work_b(idx, nmax, conf, ref_elem, path_over, path_kern, Bmat):
    """Accumulate the B matrix contribution for one training molecule.

    Args:
        idx (np.ndarray): Sparse AO start indices per reference environment and angular momentum l.
        nmax (dict[int, np.ndarray[int]]): Radial basis sizes indexed by element and l.
        conf (int): Molecule index.
        ref_elem (np.ndarray): Reference-environment atomic numbers.
        path_over (str): Template path to overlap TensorMaps.
        path_kern (str): Template path to kernel TensorMaps.
        Bmat (np.ndarray): Matrix accumulator.
    """
    over = metatensor.load(path_over.format(conf))
    k_NM = metatensor.load(path_kern.format(conf))

    for (l1, l2, q1, q2), oblock in over.items():
        msize1 = 2*l1+1
        msize2 = 2*l2+1
        nsize1 = nmax[q1][l1]
        nsize2 = nmax[q2][l2]
        kblock1 = k_NM.block(o3_lambda=l1, center_type=q1)
        kblock2 = k_NM.block(o3_lambda=l2, center_type=q2)
        oval = oblock.values.reshape(len(kblock1.samples), len(kblock2.samples), msize1, msize2, nsize1, nsize2)

        for iiref1, iref1 in enumerate(np.where(ref_elem==q1)[0]):
            for iiref2, iref2 in enumerate(np.where(ref_elem==q2)[0]):
                if iref1>iref2:
                    continue
                '''
                Non-optimized:
                ```
                dB = np.einsum('AMJ,AaMmNn,amj->NnJj', kblock1.values[...,iiref1], oval, kblock2.values[...,iiref2])
                ```
                '''
                t1 = np.einsum('AMJ,AaMmNn->amJNn', kblock1.values[...,iiref1], oval)
                dB = np.einsum('amJNn,amj->njNJ', t1, kblock2.values[...,iiref2])

                i1 = idx[iref1, l1]
                for n2 in range(nsize2):
                    for im2 in range(msize2):
                        i2 = idx[iref2, l2]+ n2*msize2 + im2
                        i12 = mpos(i1,i2)
                        if (iref1!=iref2) or (iref1==iref2 and l1<l2):
                            Bmat[i12:i12+msize1*nsize1] += dB[n2,im2,:,:].flatten()
                        elif iref1==iref2 and l1==l2:
                            i12a = i12 + msize1*n2
                            i12b = i12a + im2+1
                            Bmat[i12:i12a]  += dB[n2,im2,:n2,:].flatten()
                            Bmat[i12a:i12b] += dB[n2,im2,n2,:im2+1]


def get_b(basis, ref_elem, ntrains, trrange,
          path_over, path_kern, paths_bmat, use_mpi):
    """Build and save packed B matrices for all requested training fractions.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        ntrains (np.ndarray[int]): Cumulative training-set boundaries per fraction. The last extra one is 0.
        trrange (np.ndarray[int]): Training molecule indices.
        path_over (str): Template path to overlap TensorMaps.
        path_kern (str): Template path to kernel TensorMaps.
        paths_bmat (list[str]): Output packed-B path for each fraction.
        use_mpi (bool): Whether to use MPI.
    """
    def do_mol(imol):
        """Process one molecule index and accumulate its B contribution.

        Args:
            imol (int): Index in trrange identifying the molecule.
        """
        do_work_b(idx, basis.nmax, trrange[imol], ref_elem, path_over, path_kern, Bmat)

    totsize = basis.nao_for_mol(ref_elem)
    Bmat = np.zeros(matsize := symsize(totsize))
    idx = basis.sparse_indices(ref_elem)

    if use_mpi:
        from mpi4py import MPI  # noqa: PLC0415
        Nproc = MPI.COMM_WORLD.Get_size()
        nproc = MPI.COMM_WORLD.Get_rank()
        print_nodes(Nproc, nproc, MPI.COMM_WORLD)
        t = 0.0
        if nproc==0:
            print_mem(totsize, ntrains[-2])
            t = MPI.Wtime()
        MPI.COMM_WORLD.barrier()
    else:
        nproc = 0
        Nproc = 1

    if nproc==0:
        print_batches(ntrains, paths_bmat)
    if use_mpi:
        MPI.COMM_WORLD.barrier()

    if Nproc==1:
        for ifrac, path_bmat in enumerate(paths_bmat):
            for imol in range(ntrains[ifrac-1], ntrains[ifrac]):
                logger.info(f'{nproc:4d}: {imol:4d}', extra={'flush': True})
                do_mol(imol)
            Bmat.tofile(path_bmat)
        if use_mpi:
            t = MPI.Wtime () - t
            logger.info(f'{t=:4.2f}', extra={'flush': True})

    else:
        bufsize = min(matsize, ((1<<30)//np.array(0.0).itemsize))  # number of doubles to take 1 GiB
        div, rem = matsize//bufsize, matsize%bufsize
        if nproc==0:
            BMAT = np.zeros(bufsize)

        for ifrac, path_bmat in enumerate(paths_bmat):
            scatter_jobs(Nproc, nproc, MPI.COMM_WORLD, ntrains[ifrac-1], ntrains[ifrac], do_mol)
            MPI.COMM_WORLD.barrier()

            if nproc==0:
                tt = MPI.Wtime()
                logger.info(f'batch{ifrac}: t={tt-t:4.2f}', extra={'flush': True})
                t = tt
            for i in range(div+1):
                if (size := bufsize if i<div else rem)==0:
                    break
                MPI.COMM_WORLD.Reduce(Bmat[i*bufsize:i*bufsize+size], BMAT[:size] if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    logger.info(f'chunk #{i+1}/{div+1 if rem else div} written', extra={'flush': True})
                    with open(path_bmat, 'a' if i else 'w') as f:
                        BMAT[:size].tofile(f)
