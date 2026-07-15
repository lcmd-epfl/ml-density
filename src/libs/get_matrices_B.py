"""Compute the "B matrix" (kernel * metric matrix * kernel)."""

import logging
import numpy as np
import metatensor
from libs.get_matrices_A import print_batches, do_work_a_pitc
from libs.multi import print_nodes, scatter_jobs
from libs.pitc_lib import kmm_cholesky, molecule_lambda_inv_kmat

DEFAULT_MAX_CHUNK = 1<<30  # 1 GiB

logger = logging.getLogger('__main__')


def _build_ref_indices(ref_elem):
    """Precompute reference-environment positions grouped by element.

    Args:
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.

    Returns:
        dict[int, np.ndarray[int]]: Reference positions for each element.
    """
    return {int(q): np.where(ref_elem == q)[0] for q in np.unique(ref_elem)}


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


def do_work_b(idx, nmax, conf, ref_indices, path_over, path_kern, Bmat):
    """Accumulate the B matrix contribution for one training molecule.

    Args:
        idx (np.ndarray): Sparse AO start indices per reference environment and angular momentum l.
        nmax (dict[int, np.ndarray[int]]): Radial basis sizes indexed by element and l.
        conf (int): Molecule index.
        ref_indices (dict[int, np.ndarray[int]]): Cached reference positions grouped by element.
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

        for iiref1, iref1 in enumerate(ref_indices[q1]):
            for iiref2, iref2 in enumerate(ref_indices[q2]):
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
                i2_start = idx[iref2, l2]
                for n2 in range(nsize2):
                    for im2 in range(msize2):
                        i2 = i2_start + n2*msize2 + im2
                        i12 = mpos(i1,i2)
                        if (iref1!=iref2) or (iref1==iref2 and l1<l2):
                            Bmat[i12:i12+msize1*nsize1] += dB[n2,im2,:,:].flatten()
                        elif iref1==iref2 and l1==l2:
                            i12a = i12 + msize1*n2
                            i12b = i12a + im2+1
                            Bmat[i12:i12a]  += dB[n2,im2,:n2,:].flatten()
                            Bmat[i12a:i12b] += dB[n2,im2,n2,:im2+1]


def do_work_b_pitc(lambda_inv_i, kmat_i, Bmat):
    """Accumulate the PITC-weighted B matrix contribution for one training molecule.

    Computes dB = K_{I_i,M}^T Lambda_i^-1 K_{I_i,M}. Unlike do_work_b()'s SoR accumulation, which
    only ever touches (l,q)-block-sparse AO pairs, Lambda_i^-1 mixes across reference (l,q)
    blocks, so dB is a genuinely dense (totsize, totsize) contribution; its lower triangle is
    packed into the shared Bmat accumulator via the same mpos() convention as do_work_b().

    Args:
        lambda_inv_i (np.ndarray): Lambda_i^-1, from molecule_lambda_inv_kmat().
        kmat_i (np.ndarray): K_{I_i,M}, from molecule_lambda_inv_kmat().
        Bmat (np.ndarray): Packed lower-triangular matrix accumulator, updated in place.
    """
    dB = kmat_i.T @ lambda_inv_i @ kmat_i
    for j in range(dB.shape[0]):
        start = mpos(0, j)
        Bmat[start:start+j+1] += dB[:j+1, j]


def _accumulate_pitc(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, eta, jit, Bmat, bvec):
    """Accumulate both the B-matrix and A-vector PITC contributions for one training molecule.

    Lambda_i^-1/K_{I_i,M} are the expensive part of PITC per molecule (a dense Cholesky
    factorization/inversion each). Computing them once here for both accumulations avoids paying
    for them twice, which running get_a() and get_b() as separate passes used to require.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        mol_idx (int): Dataset index of the training molecule.
        atoms_i (np.ndarray[int]): Atomic numbers of the training molecule.
        paths (SimpleNamespace): Configured paths and path templates.
        l_mm (np.ndarray): Lower Cholesky factor of the (jittered) dense K_MM.
        eta (float): PITC noise scale.
        jit (float): Diagonal jitter added to S_i before inverting it (see molecule_lambda_inv_kmat).
        Bmat (np.ndarray): Packed lower-triangular B-matrix accumulator, updated in place.
        bvec (np.ndarray): Dense (totsize,) A-vector accumulator, updated in place.
    """
    lambda_inv_i, kmat_i, s_i, mol_i = molecule_lambda_inv_kmat(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, eta, jit)
    do_work_b_pitc(lambda_inv_i, kmat_i, Bmat)
    do_work_a_pitc(mol_idx, paths, lambda_inv_i, kmat_i, s_i, mol_i, bvec)


def _l_mm_for_pitc(basis, ref_elem, paths, o):
    """Compute the shared K_MM Cholesky factor needed by the PITC accumulation.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        paths (SimpleNamespace): Configured paths and path templates.
        o (SimpleNamespace): Configured options (reads o.full_gpr, o.jit).

    Returns:
        np.ndarray | None: l_mm (None unless o.full_gpr).
    """
    if not o.full_gpr:
        return None
    k_MM = metatensor.load(paths.kernel_mm)
    _, l_mm = kmm_cholesky(basis, ref_elem, k_MM, o.jit)
    return l_mm


def _accumulate_b(basis, ref_elem, mol_idx, atoms_i, paths, idx, ref_indices, l_mm, o, Bmat, bvec):
    """Dispatch to the PITC (B and A together) or SoR (B only) accumulation for one training molecule.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        mol_idx (int): Dataset index of the training molecule.
        atoms_i (np.ndarray[int]): Atomic numbers of the training molecule (only used for PITC).
        paths (SimpleNamespace): Configured paths and path templates.
        idx (np.ndarray): Sparse AO start indices, used by the SoR path.
        ref_indices (dict[int, np.ndarray[int]]): Cached reference positions, used by the SoR path.
        l_mm (np.ndarray | None): PITC K_MM Cholesky factor (None unless o.full_gpr).
        o (SimpleNamespace): Configured options (reads o.full_gpr, o.reg, o.jit).
        Bmat (np.ndarray): Packed lower-triangular matrix accumulator, updated in place.
        bvec (np.ndarray | None): Dense (totsize,) A-vector accumulator, updated in place
            (None unless o.full_gpr; unused by the SoR path -- get_a() builds the SoR A-vector
            separately, since it's a genuinely independent computation there, unlike for PITC).
    """
    if o.full_gpr:
        # o.reg plays the role of eta here -- see regression.py's PITC branch for why they're
        # the same symbol in the theory, not two separate regularization knobs.
        _accumulate_pitc(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, o.reg, o.jit, Bmat, bvec)
    else:
        do_work_b(idx, basis.nmax, mol_idx, ref_indices, paths.metric_matrix, paths.kernel_nm, Bmat)


def get_b(basis, ref_elem, fracs, ntrains, training_idx, paths, o, atomic_numbers, *, use_mpi):
    """Build and save packed B matrices for all requested training fractions.

    When o.full_gpr, also builds and saves the A-vector alongside the B-matrix in the same
    per-molecule pass (see _accumulate_pitc())

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        fracs (np.ndarray[float]): Training set fractions.
        ntrains (list[tuple[int], tuple[int]]): Training set boundaries
                per fraction batch corresponding to the new molecules wrt the previous batch.
        training_idx (np.ndarray[int]): Training molecule indices.
        paths (SimpleNamespace): Configured paths and path templates..
        o (SimpleNamespace): Configured options (reads o.full_gpr, o.reg, o.jit).
        atomic_numbers (np.ndarray): Per-molecule atomic-number arrays, indexed by dataset index.
        use_mpi (bool): Whether to use MPI.
    """
    def do_mol(imol):
        """Process one molecule index and accumulate its B contribution.

        Args:
            imol (int): Index in training_idx identifying the molecule.
        """
        mol_idx = training_idx[imol]
        _accumulate_b(basis, ref_elem, mol_idx, atomic_numbers[mol_idx], paths, idx, ref_indices, l_mm, o, Bmat, bvec)

    totsize = basis.nao_for_mol(ref_elem)
    Bmat = np.zeros(matsize := symsize(totsize))
    bvec = np.zeros(totsize) if o.full_gpr else None
    idx = basis.sparse_indices(ref_elem)
    ref_indices = _build_ref_indices(ref_elem)
    l_mm = _l_mm_for_pitc(basis, ref_elem, paths, o)

    if use_mpi:
        from mpi4py import MPI  # noqa: PLC0415
        Nproc = MPI.COMM_WORLD.Get_size()
        nproc = MPI.COMM_WORLD.Get_rank()
        print_nodes(Nproc, nproc, MPI.COMM_WORLD)
        t = 0.0
        if nproc==0:
            print_mem(totsize, ntrains[-1][1])
            t = MPI.Wtime()
        MPI.COMM_WORLD.barrier()
    else:
        nproc = 0
        Nproc = 1

    if nproc==0:
        print_batches(fracs, ntrains, paths.bmat)
        if o.full_gpr:
            print_batches(fracs, ntrains, paths.avec)
    if use_mpi:
        MPI.COMM_WORLD.barrier()

    if Nproc==1:
        for frac, ntrain in zip(fracs, ntrains, strict=True):
            for imol in range(ntrain[0], ntrain[1]):
                logger.info(f'{nproc:4d}: {imol:4d}', extra={'flush': True})
                do_mol(imol)
            Bmat.tofile(paths.bmat.format(train_frac=frac))
            if o.full_gpr:
                np.savetxt(paths.avec.format(train_frac=frac), bvec)
        if use_mpi:
            t = MPI.Wtime () - t
            logger.info(f'{t=:4.2f}', extra={'flush': True})

    else:

        bufsize = min(matsize, (DEFAULT_MAX_CHUNK//np.array(0.0).itemsize))  # number of doubles in a max. chunk
        div, rem = matsize//bufsize, matsize%bufsize
        if nproc==0:
            BMAT = np.zeros(bufsize)
        if o.full_gpr and nproc==0:
            BVEC = np.zeros(totsize)

        for ifrac, (frac, ntrain) in enumerate(zip(fracs, ntrains, strict=True)):
            scatter_jobs(Nproc, nproc, MPI.COMM_WORLD, ntrain[0], ntrain[1], do_mol)
            MPI.COMM_WORLD.barrier()

            if nproc==0:
                tt = MPI.Wtime()
                logger.info(f'batch{ifrac}: t={tt-t:4.2f}', extra={'flush': True})
                t = tt
            if o.full_gpr:
                MPI.COMM_WORLD.Reduce(bvec, BVEC if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    np.savetxt(paths.avec.format(train_frac=frac), BVEC)
            for i in range(div+1):
                if (size := bufsize if i<div else rem)==0:
                    break
                MPI.COMM_WORLD.Reduce(Bmat[i*bufsize:i*bufsize+size], BMAT[:size] if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    logger.info(f'chunk #{i+1}/{div+1 if rem else div} written', extra={'flush': True})
                    with open(paths.bmat.format(train_frac=frac), 'a' if i else 'w') as f:
                        BMAT[:size].tofile(f)
