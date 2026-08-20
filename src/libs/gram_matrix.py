"""Compute the Gram matrix (kernel^T * metric matrix * kernel).

The Gram matrix of the reference-environment kernel columns under the metric-weighted inner
product: sum_i K_{I_i,M}^T M_i K_{I_i,M} (SA-GPR and gpr_DTC) or
sum_i K_{I_i,M}^T Lambda_i^-1 K_{I_i,M} (gpr_PITC).
"""

import functools
import logging
import numpy as np
import scipy.linalg as spl
import metatensor
from libs.target_vector import print_batches, do_work_target_pitc
from libs.multi import print_nodes, scatter_jobs
from libs.gp_common import kmm_cholesky
from libs.pitc_lib import molecule_lambda_chol_knm
from libs.tmap import tmap2averages
from libs.config_utils import GPR_DTC, GPR_PITC

# Buffer for the final MPI Reduce of the packed Gram onto rank 0 (communication phase, rank 0 only; larger just means fewer messages).
DEFAULT_MAX_CHUNK = 1<<30  # 1 GiB
# Cap on the transient per-molecule d_gram block in do_work_gram_pitc (assembly phase, every rank;
# smaller keeps the per-rank memory peak low).
DEFAULT_GRAM_CHUNK_BYTES = 256<<20  # 256 MiB
# See _gram_chunk_cols() for explanations.
MIN_EFFICIENT_GRAM_CHUNK_COLS = 128

logger = logging.getLogger('__main__')


def _build_ref_indices(ref_elem):
    """Precompute reference-environment positions grouped by element.

    Args:
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.

    Returns:
        dict[int, np.ndarray[int]]: Reference positions for each element.
    """
    return {int(q): np.where(ref_elem == q)[0] for q in np.unique(ref_elem)}


def print_mem(nao_ref, ntrain):
    """Log estimated memory usage to store the result.

    Args:
        nao_ref (int): Problem dimensionality (number of AO coefficients).
        ntrain (int): Number of training molecules.
    """
    b2mib = 1.0/(1<<20)
    b2gib = 1.0/(1<<30)
    size = symsize(nao_ref)*np.array(0.0).itemsize
    logger.info(f"""Problem dimensionality = {nao_ref}\n\
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


def do_work_gram(idx, nmax, conf, ref_indices, path_metric, path_kern, gram_mat):
    """Accumulate the Gram-matrix contribution for one training molecule.

    Args:
        idx (np.ndarray): Sparse AO start indices per reference environment and angular momentum l.
        nmax (dict[int, np.ndarray[int]]): Radial basis sizes indexed by element and l.
        conf (int): Molecule index.
        ref_indices (dict[int, np.ndarray[int]]): Cached reference positions grouped by element.
        path_metric (str): Template path to metric-matrix TensorMaps.
        path_kern (str): Template path to kernel TensorMaps.
        gram_mat (np.ndarray): Matrix accumulator.
    """
    metric = metatensor.load(path_metric.format(conf))
    k_NM = metatensor.load(path_kern.format(conf))

    for (l1, l2, q1, q2), mblock in metric.items():
        msize1 = 2*l1+1
        msize2 = 2*l2+1
        nsize1 = nmax[q1][l1]
        nsize2 = nmax[q2][l2]
        kblock1 = k_NM.block(o3_lambda=l1, center_type=q1)
        kblock2 = k_NM.block(o3_lambda=l2, center_type=q2)
        mval = mblock.values.reshape(len(kblock1.samples), len(kblock2.samples), msize1, msize2, nsize1, nsize2)

        for iiref1, iref1 in enumerate(ref_indices[q1]):
            for iiref2, iref2 in enumerate(ref_indices[q2]):
                if iref1>iref2:
                    continue
                '''
                Non-optimized:
                ```
                d_gram = np.einsum('AMJ,AaMmNn,amj->NnJj', kblock1.values[...,iiref1], mval, kblock2.values[...,iiref2])
                ```
                '''
                t1 = np.einsum('AMJ,AaMmNn->amJNn', kblock1.values[...,iiref1], mval)
                d_gram = np.einsum('amJNn,amj->njNJ', t1, kblock2.values[...,iiref2])

                i1 = idx[iref1, l1]
                i2_start = idx[iref2, l2]
                for n2 in range(nsize2):
                    for im2 in range(msize2):
                        i2 = i2_start + n2*msize2 + im2
                        i12 = mpos(i1,i2)
                        if (iref1!=iref2) or (iref1==iref2 and l1<l2):
                            gram_mat[i12:i12+msize1*nsize1] += d_gram[n2,im2,:,:].flatten()
                        elif iref1==iref2 and l1==l2:
                            i12a = i12 + msize1*n2
                            i12b = i12a + im2+1
                            gram_mat[i12:i12a]  += d_gram[n2,im2,:n2,:].flatten()
                            gram_mat[i12a:i12b] += d_gram[n2,im2,n2,:im2+1]


@functools.lru_cache(maxsize=None)
def _gram_chunk_cols(nao_ref, chunk_bytes, itemsize):
    """Column-chunk width for do_work_gram_pitc, capped so the (nao_ref, c) temporary stays under chunk_bytes.

    Cached so the small-chunk efficiency warning below is emitted at most once per
    (nao_ref, chunk_bytes) rather than once per molecule (do_work_gram_pitc is called N times).

    Args:
        nao_ref (int): Problem dimensionality (Gram-matrix side length).
        chunk_bytes (int): Upper bound on the per-chunk dense temporary.
        itemsize (int): Bytes per matrix element (8 for float64).

    Returns:
        int: Number of columns c per chunk, in [1, nao_ref].
    """
    c = max(1, min(nao_ref, chunk_bytes // (nao_ref * itemsize)))
    if c < MIN_EFFICIENT_GRAM_CHUNK_COLS:
        logger.warning(
            f'PITC Gram chunk width is {c} columns (< {MIN_EFFICIENT_GRAM_CHUNK_COLS}) for nao_ref={nao_ref}, '
            f'chunk_bytes={chunk_bytes} ({chunk_bytes>>20} MiB). At this width the per-molecule chunk matmul '
            f'(nao_ref, nao_i) @ (nao_i, c) is narrow and memory-bound (arithmetic intensity ~ c), so BLAS runs '
            f'well below peak, and the chunk count (nao_ref/c) and its Python scatter overhead grow -- Gram '
            f'assembly gets slow. Raise DEFAULT_GRAM_CHUNK_BYTES to widen the chunk (trades more peak memory).',
            extra={'flush': True},
        )
    return c


def do_work_gram_pitc(v_i, gram_mat, chunk_bytes=DEFAULT_GRAM_CHUNK_BYTES):
    """Accumulate the PITC-weighted Gram-matrix contribution for one training molecule.

    Adds d_gram = K_{I_i,M}^T Lambda_i^-1 K_{I_i,M} into the packed lower-triangular accumulator
    without forming the full nao_ref^2 matrix: the per-chunk temporary is bounded by chunk_bytes
    (independent of nao_ref). See docs/training_complexity.md.

    v_i^T v_i ensure symmetry by construction on the contrary of other methods.

    Args:
        v_i (np.ndarray): L_i^-1 K_{I_i,M} (nao_i, nao_ref), with L_i L_i^T = Lambda_i.
        gram_mat (np.ndarray): Packed lower-triangular matrix accumulator, updated in place.
        chunk_bytes (int): Upper bound on the per-chunk dense temporary (default DEFAULT_GRAM_CHUNK_BYTES).
    """
    nao_ref = v_i.shape[1]
    v_i_t = v_i.T # (nao_ref, nao_i)
    # Column-chunk width so the worst-case (nao_ref, c) temporary stays under chunk_bytes
    c = _gram_chunk_cols(nao_ref, chunk_bytes, v_i.itemsize)
    for j0 in range(0, nao_ref, c):
        j1 = min(j0 + c, nao_ref)
        block = v_i_t[:j1] @ v_i[:, j0:j1]   # (j1, j1-j0); rows > j are discarded on scatter
        for j in range(j0, j1):
            start = mpos(0, j)
            gram_mat[start:start+j+1] += block[:j+1, j-j0]


def _accumulate_pitc(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, av_coefs, eta, jit, gram_mat, target_vec, ml_terms):
    """Accumulate both the Gram-matrix and target-vector PITC contributions for one training molecule.

    Lambda_i/K_{I_i,M} are the expensive part of PITC per molecule (a dense Cholesky factorization
    each). Computing them once here for both accumulations avoids paying for them twice, which
    running get_target_vector() and get_gram_matrix() as separate passes used to require. The
    half-solve v_i = L_i^-1 K_{I_i,M} is shared for the same reason: the Gram needs v_i^T v_i and
    the target vector needs v_i^T (L_i^-1 y_i), so one triangular solve serves both.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        mol_idx (int): Dataset index of the training molecule.
        atoms_i (np.ndarray[int]): Atomic numbers of the training molecule.
        paths (SimpleNamespace): Configured paths and path templates.
        l_mm (np.ndarray): Lower Cholesky factor of the (jittered) dense K_MM.
        av_coefs (dict[int, np.ndarray]): Per-element l=0 averages (see do_work_target_pitc).
        eta (float): PITC noise scale.
        jit (float): Relative diagonal jitter (see molecule_lambda_chol_knm).
        gram_mat (np.ndarray): Packed lower-triangular Gram-matrix accumulator, updated in place.
        target_vec (np.ndarray): Dense (nao_ref,) target-vector accumulator, updated in place.
        ml_terms (np.ndarray): Dense (2,) marginal-likelihood accumulator, updated in place
            (see do_work_target_pitc).
    """
    l_lambda_i, k_nm_i = molecule_lambda_chol_knm(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, eta, jit)
    v_i = spl.solve_triangular(l_lambda_i, k_nm_i, lower=True)   # L_i^-1 K_{I_i,M}
    do_work_gram_pitc(v_i, gram_mat)
    do_work_target_pitc(mol_idx, paths, basis, atoms_i, av_coefs, l_lambda_i, v_i, target_vec, ml_terms)


def _l_mm_for_pitc(basis, ref_elem, paths, o):
    """Compute the shared K_MM Cholesky factor needed by the PITC accumulation.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        paths (SimpleNamespace): Configured paths and path templates.
        o (SimpleNamespace): Configured options (reads o.regression_model, o.jit).

    Returns:
        np.ndarray | None: l_mm (None unless o.regression_model==GPR_PITC; gpr_DTC and SA-GPR never need K_MM during
        the assembly, only in regression.py).
    """
    if o.regression_model!=GPR_PITC:
        return None
    k_MM = metatensor.load(paths.kernel_mm)
    _, l_mm = kmm_cholesky(basis, ref_elem, k_MM, o.jit)
    return l_mm


def _l_mm_shared(basis, ref_elem, paths, o, comm):
    """Build the K_MM Cholesky factor once per node in MPI shared memory.

    Unlike _l_mm_for_pitc (one private copy per rank), this allocates a single (nao_ref, nao_ref)
    array per shared-memory node via MPI.Win.Allocate_shared, builds l_mm on that node's rank 0, and
    returns a view every rank on the node shares. molecule_lambda_chol_knm only reads l_mm (through
    solve_triangular), so concurrent access is safe with no locking. Cuts the resident l_mm footprint from
    ~nao_ref^2 per rank to ~nao_ref^2 per node, and loads/factorizes K_MM once per node instead of
    once per rank.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        paths (SimpleNamespace): Configured paths and path templates.
        o (SimpleNamespace): Configured options (reads o.regression_model, o.jit).
        comm (mpi4py.MPI.Comm): Communicator to split into shared-memory nodes.

    Returns:
        tuple[np.ndarray | None, mpi4py.MPI.Win | None]: the shared l_mm view and its window (both
        None unless o.regression_model==GPR_PITC). Keep the window referenced while l_mm is in use and Free()
        it after.
    """
    if o.regression_model!=GPR_PITC:
        return None, None
    from mpi4py import MPI  # noqa: PLC0415
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    nao_ref = basis.nao_for_mol(ref_elem)
    itemsize = np.dtype(np.float64).itemsize
    nbytes = nao_ref * nao_ref * itemsize if node_comm.Get_rank() == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    buf, _ = win.Shared_query(0)
    l_mm = np.ndarray(buffer=buf, dtype=np.float64, shape=(nao_ref, nao_ref))
    if node_comm.Get_rank() == 0:
        k_MM = metatensor.load(paths.kernel_mm)
        _, l_mm_local = kmm_cholesky(basis, ref_elem, k_MM, o.jit)
        l_mm[:] = l_mm_local  # publish into the shared segment
    node_comm.Barrier()  # ensure the factor is visible to every rank before it is read
    return l_mm, win


def _accumulate_gram(basis, ref_elem, mol_idx, atoms_i, paths, idx, ref_indices, l_mm, av_coefs, coef_norms, o, gram_mat, target_vec, ml_terms):
    """Dispatch to the gpr_PITC (Gram and target together) or SA-GPR/gpr_DTC (Gram only) accumulation for one training molecule.

    gpr_DTC shares the SA-GPR Gram exactly (Lambda^-1 = M/eta), so it takes the same branch and only adds
    its marginal-likelihood contribution on top -- both terms of which are already known: the
    quadratic form y_i^T M_i y_i was tabulated by preprocess.py, and the count is the molecule's
    number of AOs.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        mol_idx (int): Dataset index of the training molecule.
        atoms_i (np.ndarray[int]): Atomic numbers of the training molecule (only used for PITC/DTC).
        paths (SimpleNamespace): Configured paths and path templates.
        idx (np.ndarray): Sparse AO start indices, used by the SA-GPR/gpr_DTC path.
        ref_indices (dict[int, np.ndarray[int]]): Cached reference positions, used by the SA-GPR/gpr_DTC path.
        l_mm (np.ndarray | None): PITC K_MM Cholesky factor (None unless o.regression_model==GPR_PITC).
        av_coefs (dict[int, np.ndarray] | None): Per-element l=0 averages, used by the PITC target
            (None unless o.regression_model==GPR_PITC).
        coef_norms (np.ndarray | None): p.coef_norms table, column 1 holding y_i^T M_i y_i per
            molecule (None unless o.regression_model==GPR_DTC).
        o (SimpleNamespace): Configured options (reads o.regression_model, o.reg, o.jit).
        gram_mat (np.ndarray): Packed lower-triangular matrix accumulator, updated in place.
        target_vec (np.ndarray | None): Dense (nao_ref,) target-vector accumulator, updated in place
            (None unless o.regression_model==GPR_PITC; unused by the SA-GPR/gpr_DTC path -- get_target_vector()
            builds their target vector separately, since it's a genuinely independent computation
            there, unlike for PITC).
        ml_terms (np.ndarray | None): Dense (2,) marginal-likelihood accumulator, updated in place
            (None unless o.is_gpr; see do_work_target_pitc).
    """
    if o.regression_model==GPR_PITC:
        # o.reg plays the role of eta here -- see regression.py's PITC branch for why they're
        # the same symbol in the theory, not two separate regularization knobs.
        _accumulate_pitc(basis, ref_elem, mol_idx, atoms_i, paths, l_mm, av_coefs, o.reg, o.jit, gram_mat, target_vec, ml_terms)
        return
    do_work_gram(idx, basis.nmax, mol_idx, ref_indices, paths.metric_matrix, paths.kernel_nm, gram_mat)
    if o.regression_model==GPR_DTC:
        # (c~_T - c_av)^T M_T (c~_T - c_av) and N_T = sum_i nao_i, the two data-side inputs to
        # main.pdf Eq. 26. The 1/lambda it carries is applied once in regression.py, not per
        # molecule (see dtc_lib.fit_prior_scale).
        ml_terms[0] += coef_norms[mol_idx, 1]
        ml_terms[1] += basis.nao_for_mol(atoms_i)


def get_gram_matrix(basis, ref_elem, fracs, ntrains, training_idx, paths, o, atomic_numbers, *, use_mpi):
    """Build and save packed Gram matrices for all requested training fractions.

    Under o.regression_model==GPR_PITC, also builds and saves the target vector alongside the Gram matrix in
    the same per-molecule pass (see _accumulate_pitc()); SA-GPR and gpr_DTC build theirs in
    target_vector.get_target_vector() instead. Both GP models additionally save the
    marginal-likelihood accumulators consumed by regression.py's sigma_p^2 fit.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        ref_elem (np.ndarray[int]): Reference-environment atomic numbers.
        fracs (np.ndarray[float]): Training set fractions.
        ntrains (list[tuple[int], tuple[int]]): Training set boundaries
                per fraction batch corresponding to the new molecules wrt the previous batch.
        training_idx (np.ndarray[int]): Training molecule indices.
        paths (SimpleNamespace): Configured paths and path templates..
        o (SimpleNamespace): Configured options (reads o.regression_model, o.reg, o.jit).
        atomic_numbers (np.ndarray): Per-molecule atomic-number arrays, indexed by dataset index.
        use_mpi (bool): Whether to use MPI.
    """
    def do_mol(imol):
        """Process one molecule index and accumulate its Gram-matrix contribution.

        Args:
            imol (int): Index in training_idx identifying the molecule.
        """
        mol_idx = training_idx[imol]
        _accumulate_gram(basis, ref_elem, mol_idx, atomic_numbers[mol_idx], paths, idx, ref_indices, l_mm, av_coefs, coef_norms, o, gram_mat, target_vec, ml_terms)

    nao_ref = basis.nao_for_mol(ref_elem)
    gram_mat = np.zeros(matsize := symsize(nao_ref))
    # Only gpr_PITC builds its target vector here; SA-GPR and gpr_DTC get theirs from get_target_vector().
    target_vec = np.zeros(nao_ref) if o.regression_model==GPR_PITC else None
    av_coefs = tmap2averages(metatensor.load(paths.spherical_averages)) if o.regression_model==GPR_PITC else None
    coef_norms = np.load(paths.coef_norms) if o.regression_model==GPR_DTC else None
    # ml_terms feeds the sigma_p^2 fit, which both GP models do.
    ml_terms = np.zeros(2) if o.is_gpr else None
    idx = basis.sparse_indices(ref_elem)
    ref_indices = _build_ref_indices(ref_elem)
    l_mm_win = None

    if use_mpi:
        from mpi4py import MPI  # noqa: PLC0415
        Nproc = MPI.COMM_WORLD.Get_size()
        nproc = MPI.COMM_WORLD.Get_rank()
        print_nodes(Nproc, nproc, MPI.COMM_WORLD)
        # One K_MM Cholesky factor per node in shared memory (see _l_mm_shared), not one per rank.
        l_mm, l_mm_win = _l_mm_shared(basis, ref_elem, paths, o, MPI.COMM_WORLD)
        t = 0.0
        if nproc==0:
            print_mem(nao_ref, ntrains[-1][1])
            t = MPI.Wtime()
        MPI.COMM_WORLD.barrier()
    else:
        nproc = 0
        Nproc = 1
        l_mm = _l_mm_for_pitc(basis, ref_elem, paths, o)

    if nproc==0:
        print_batches(fracs, ntrains, paths.gram_mat)
        if target_vec is not None:
            print_batches(fracs, ntrains, paths.target_vec)
    if use_mpi:
        MPI.COMM_WORLD.barrier()

    if Nproc==1:
        for frac, ntrain in zip(fracs, ntrains, strict=True):
            for imol in range(ntrain[0], ntrain[1]):
                logger.info(f'{nproc:4d}: {imol:4d}', extra={'flush': True})
                do_mol(imol)
            gram_mat.tofile(paths.gram_mat.format(train_frac=frac))
            if target_vec is not None:
                np.savetxt(paths.target_vec.format(train_frac=frac), target_vec)
            if o.is_gpr:
                np.savetxt(paths.ml_terms.format(train_frac=frac), ml_terms)
        if use_mpi:
            t = MPI.Wtime () - t
            logger.info(f'{t=:4.2f}', extra={'flush': True})

    else:

        bufsize = min(matsize, (DEFAULT_MAX_CHUNK//np.array(0.0).itemsize))  # number of doubles in a max. chunk
        div, rem = matsize//bufsize, matsize%bufsize
        if nproc==0:
            GRAM_MAT = np.zeros(bufsize)
        if o.is_gpr and nproc==0:
            TARGET_VEC = np.zeros(nao_ref)
            ML_TERMS = np.zeros(2)

        for ifrac, (frac, ntrain) in enumerate(zip(fracs, ntrains, strict=True)):
            scatter_jobs(Nproc, nproc, MPI.COMM_WORLD, ntrain[0], ntrain[1], do_mol)
            MPI.COMM_WORLD.barrier()

            if nproc==0:
                tt = MPI.Wtime()
                logger.info(f'batch{ifrac}: t={tt-t:4.2f}', extra={'flush': True})
                t = tt
            if target_vec is not None:
                MPI.COMM_WORLD.Reduce(target_vec, TARGET_VEC if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    np.savetxt(paths.target_vec.format(train_frac=frac), TARGET_VEC)
            if o.is_gpr:
                MPI.COMM_WORLD.Reduce(ml_terms, ML_TERMS if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    np.savetxt(paths.ml_terms.format(train_frac=frac), ML_TERMS)
            for i in range(div+1):
                if (size := bufsize if i<div else rem)==0:
                    break
                MPI.COMM_WORLD.Reduce(gram_mat[i*bufsize:i*bufsize+size], GRAM_MAT[:size] if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    logger.info(f'chunk #{i+1}/{div+1 if rem else div} written', extra={'flush': True})
                    with open(paths.gram_mat.format(train_frac=frac), 'a' if i else 'w') as f:
                        GRAM_MAT[:size].tofile(f)

    if l_mm_win is not None:  # release the shared K_MM window (collective over each node)
        l_mm_win.Free()
