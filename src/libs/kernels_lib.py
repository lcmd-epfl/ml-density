"""Compute kernels with reference environments."""


import numpy as np
import metatensor
from libs.tmap import kernels2tmap, kmm2tmap


def kernel_nm(atoms, soap, soap_ref, imol=0):
    """Compute kernel between one molecule and references.

    Args:
        atoms (np.ndarray[int]): Atomic numbers of the query molecule.
        soap (metatensor.TensorMap): λ-SOAP power spectrum of the query molecule.
        soap_ref (metatensor.TensorMap): Reference λ-SOAP power spectra.
        imol (int): Sample index used in TensorMap sample labels for the query molecule.

    Returns:
        metatensor.TensorMap: Kernel TensorMap indexed by (o3_lambda, center_type).
    """
    keys1 = {tuple(key) for key in soap.keys}
    keys2 = {tuple(key) for key in soap_ref.keys}
    keys  = sorted(keys1 & keys2, key=lambda x: x[::-1])
    kernel = {key: [] for key in keys}

    for iat, q in enumerate(atoms):
        for (l, q_) in keys:
            if q_!=q:
                continue
            block = soap.block(o3_lambda=l, center_type=q)
            isamp = block.samples.position((imol, iat))
            vals  = block.values[isamp,:,:]
            block_ref = soap_ref.block(o3_lambda=l, center_type=q)
            vals_ref  = block_ref.values
            pre_kernel = np.einsum('rmx,Mx->rMm', vals_ref, vals)
            # Normalize with zeta=2
            if l==0:
                factor = pre_kernel
            kernel[l,q].append(pre_kernel * factor)
    return kernels2tmap(atoms, kernel)


def kernel_for_mol(atoms, power_ref, power_file, kernel_file):
    """Wrap kernel_nm().

    Args:
        atoms (np.ndarray | list[np.ndarray]): Atom numbers for each molecule.
        power_ref (metatensor.TensorMap): Reference λ-SOAP power spectra.
        power_file (str): Path to molecule λ-SOAP power spectrum file.
        kernel_file (str): Output path.
    """
    power = metatensor.load(power_file)
    k_NM = kernel_nm(atoms, power, power_ref)
    metatensor.save(f'{kernel_file}', k_NM)


def kernel_block_to_dense_rect(basis, atoms_left, elem_right, k_tmap):
    """Expand a K_NM-style kernel TensorMap into a dense (nao_left, nao_right) matrix.

    A kernel value depends only on (l, q), not on the radial channel n. Each radial channel carries
    its own weight, so the model contracts channels *diagonally*: predicted coeff (atom, n, l, m) =
    sum_{ref, m'} k_l(atom, ref)[m, m'] w(ref, n, l, m'), with output n equal to the weight's n.
    The dense block is therefore block-diagonal in n (np.kron(eye(nsize), dk)), matching both the
    prediction einsum ('mMr,rMn->mn') and regression.fill_matrix()'s diagonal-in-n placement. A full
    np.tile(dk, (nsize, nsize)) broadcast would instead couple every (n_left, n_right) pair, which is
    a different, rank-deficient operator (rank msize per block, ~nsize too large) inconsistent with
    prediction -- the cause of the earlier full_gpr fit failure.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        atoms_left (np.ndarray[int]): Atomic numbers of the molecule side (rows), natural atom order.
        elem_right (np.ndarray[int]): Atomic numbers of the reference-like side (columns), in the
            order used to build k_tmap's reference axis (e.g. ref_elem for K_NM/K_N*M).
        k_tmap (metatensor.TensorMap): Kernel TensorMap keyed by (o3_lambda, center_type), blocks
            shaped (n_atoms_of_q_left, msize, msize, n_atoms_of_q_right), as produced by kernel_nm().

    Returns:
        np.ndarray: Dense (nao_left, nao_right) kernel matrix.
    """
    idx_left  = basis.sparse_indices(atoms_left)
    idx_right = basis.sparse_indices(elem_right)
    dense = np.zeros((basis.nao_for_mol(atoms_left), basis.nao_for_mol(elem_right)))
    for (l, q), kblock in k_tmap.items():
        msize = 2*l+1
        nsize = basis.nmax[q][l]
        for iiat, iat in enumerate(np.where(atoms_left==q)[0]):
            i1 = idx_left[iat, l]
            for iiref, iref in enumerate(np.where(elem_right==q)[0]):
                i2 = idx_right[iref, l]
                dk = kblock.values[iiat, :, :, iiref]
                dense[i1:i1+nsize*msize, i2:i2+nsize*msize] = np.kron(np.eye(nsize), dk)
    return dense


def kernel_block_to_dense_self(basis, elem, k_mm_tmap):
    """Expand a K_MM-style self/pairwise kernel TensorMap into a dense (nao, nao) matrix.

    Args:
        basis (.functions.Basis): Basis used for AO indexing.
        elem (np.ndarray[int]): Atomic numbers defining the AO layout on both sides, in the order
            matching k_mm_tmap's ref_env1/ref_env2 sample axis (e.g. ref_elem for K_MM, or one
            molecule's own natural atom order for K_{I_i,I_i} / K_{N*,N*}).
        k_mm_tmap (metatensor.TensorMap): TensorMap produced by kernel_mm().

    Returns:
        np.ndarray: Dense symmetric (nao, nao) block-diagonal kernel matrix, full (n1, n2) broadcast.
    """
    idx = basis.sparse_indices(elem)
    dense = np.zeros((basis.nao_for_mol(elem), basis.nao_for_mol(elem)))
    for (l, q), kblock in k_mm_tmap.items():
        msize = 2*l+1
        nsize = basis.nmax[q][l]
        atoms_of_q = np.where(elem==q)[0]
        nsamp = len(atoms_of_q)
        values = kblock.values[...,0].reshape(nsamp, nsamp, msize, msize)
        for iia, ia in enumerate(atoms_of_q):
            i1 = idx[ia, l]
            for iib, ib in enumerate(atoms_of_q):
                i2 = idx[ib, l]
                dk = values[iia, iib]
                dense[i1:i1+nsize*msize, i2:i2+nsize*msize] = np.kron(np.eye(nsize), dk)
    return dense


def kernel_mm(lmax, power_ref):
    """Compute reference-reference kernel.

    Args:
        lmax (dict[int, int]): Maximum angular momentum for each center type.
        power_ref (metatensor.TensorMap): Reference power spectra.

    Returns:
        metatensor.TensorMap: Kernel TensorMap over all reference-environment pairs.
    """
    samples = {}
    k_MM = {}
    for (l, q), rblock in power_ref.items():
        msize = 2*l+1
        nsamp = len(rblock.samples)
        if q not in samples:
            samples[q] = list(rblock.samples)
        k_MM[l,q] = np.zeros((nsamp, nsamp, msize, msize))
        for iiref1 in range(nsamp):
            vec1 = rblock.values[iiref1]
            for iiref2 in range(iiref1, nsamp):
                vec2 = rblock.values[iiref2]
                dot = vec1 @ vec2.T
                k_MM[l, q][iiref1, iiref2] = dot
                if iiref1!=iiref2:
                    k_MM[l,q][iiref2, iiref1] = dot.T
    for q, lm in lmax.items():
        # Mind the descending order of l
        for l in range(lm, -1, -1):
            k_MM[l,q] *= k_MM[0,q]

    return kmm2tmap(samples, k_MM)
