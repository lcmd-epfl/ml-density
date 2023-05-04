import numpy as np
import equistore
from libs.tmap import kernels2tmap


def kernel_nm_sparse_indices(lmax, ref_elements, atomic_numbers):
    llmax = max(lmax.values())
    kernel_sparse_indices = np.zeros((len(ref_elements), len(atomic_numbers), llmax+1), dtype=int)
    kernel_size = 0
    for iref, q in enumerate(ref_elements):
        nq  = np.count_nonzero(atomic_numbers==q)
        for l in range(lmax[q]+1):
            msize = 2*l+1
            for iat in range(nq):
                kernel_sparse_indices[iref,iat,l] = kernel_size
                kernel_size += msize*msize
    return kernel_size, kernel_sparse_indices


def kernel_nm(atom_charges, soap, soap_ref, imol=0):
    keys1 = set([tuple(key) for key in soap.keys])
    keys2 = set([tuple(key) for key in soap_ref.keys])
    keys  = sorted(keys1 & keys2, key=lambda x: x[::-1])
    kernel = {key: [] for key in keys}

    for iat, q in enumerate(atom_charges):
        for (l,q_) in keys:
            if q_!=q: continue
            block = soap.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((imol, iat))
            vals  = block.values[isamp,:,:]
            block_ref = soap_ref.block(spherical_harmonics_l=l, species_center=q)
            vals_ref  = block_ref.values
            pre_kernel = np.einsum('rmx,Mx->rMm', vals_ref, vals)
            # Normalize with zeta=2
            if l==0:
                factor = pre_kernel
            kernel[(l,q)].append(pre_kernel * factor)
    kernel = kernels2tmap(atom_charges, kernel)
    return kernel


def kernel_nm_flatten(kernel_size, kernel_sparse_indices,
                      ref_elements, atomic_numbers, k_NM, imol=0):

    k_NM_flat = np.zeros(kernel_size)
    for (l, q) in k_NM.keys:
        nq = np.count_nonzero(atomic_numbers==q)
        msize = 2*l+1
        kblock = k_NM.block(spherical_harmonics_l=l, species_center=q)
        for iiref, iref in enumerate(np.where(ref_elements==q)[0]):
            for iatq in range(nq):
                ik = kernel_sparse_indices[iref,iatq,l]
                k_NM_flat[ik:ik+msize*msize] = kblock.values[iatq,:,:,iiref].T.flatten()
    return k_NM_flat


def kernel_for_mol(lmax, ref_elements, atomic_numbers, power_ref, power_file, kernel_file):

    power = equistore.load(power_file)
    k_NM = kernel_nm(atomic_numbers, power, power_ref)
    equistore.save(f'{kernel_file}.npz', k_NM)

    kernel_size, kernel_sparse_indices = kernel_nm_sparse_indices(lmax, ref_elements, atomic_numbers)
    k_NM_flat = kernel_nm_flatten(kernel_size, kernel_sparse_indices, ref_elements, atomic_numbers, k_NM)
    np.savetxt(kernel_file, k_NM_flat)


def kernel_mm(M, powerrefbase, ref_elements):

    power_ref = equistore.load(f'{powerrefbase}_{M}.npz')
    llmax = max(l for l, q in power_ref.keys)
    k_MM = np.zeros((llmax+1, M*(2*llmax+1), M*(2*llmax+1)))

    for (l, q), rblock in power_ref:
        msize = 2*l+1
        for iref1 in rblock.samples:
            pos1 = rblock.samples.position(iref1)
            vec1 = rblock.values[pos1]
            for iref2 in rblock.samples:
                pos2 = rblock.samples.position(iref2)
                vec2 = rblock.values[pos2]
                k_MM[l, iref1[0]*msize:(iref1[0]+1)*msize, iref2[0]*msize:(iref2[0]+1)*msize] = vec1 @ vec2.T

    for l in range(llmax, -1, -1):
        msize = 2*l+1
        for iref1 in range(M):
            for iref2 in range(M):
                k_MM[l, iref1*msize:(iref1+1)*msize, iref2*msize:(iref2+1)*msize] *= k_MM[0, iref1, iref2]

    return k_MM
