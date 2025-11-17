import numpy as np
import metatensor
from libs.tmap import kernels2tmap, kmm2tmap


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
    keys1 = {tuple(key) for key in soap.keys}
    keys2 = {tuple(key) for key in soap_ref.keys}
    keys  = sorted(keys1 & keys2, key=lambda x: x[::-1])
    kernel = {key: [] for key in keys}

    for iat, q in enumerate(atom_charges):
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
            kernel[(l,q)].append(pre_kernel * factor)
    kernel = kernels2tmap(atom_charges, kernel)
    return kernel


def kernel_nm_flatten(kernel_size, kernel_sparse_indices,
                      ref_elements, atomic_numbers, k_NM):

    k_NM_flat = np.zeros(kernel_size)
    for (l, q) in k_NM.keys:
        nq = np.count_nonzero(atomic_numbers==q)
        msize = 2*l+1
        kblock = k_NM.block(o3_lambda=l, center_type=q)
        for iiref, iref in enumerate(np.where(ref_elements==q)[0]):
            for iatq in range(nq):
                ik = kernel_sparse_indices[iref,iatq,l]
                k_NM_flat[ik:ik+msize*msize] = kblock.values[iatq,:,:,iiref].T.flatten()
    return k_NM_flat


def kernel_for_mol(lmax, ref_elements, atomic_numbers, power_ref, power_file, kernel_file, save_txt=False):
    power = metatensor.load(power_file)
    k_NM = kernel_nm(atomic_numbers, power, power_ref)
    metatensor.save(f'{kernel_file}', k_NM)
    if save_txt:
        kernel_size, kernel_sparse_indices = kernel_nm_sparse_indices(lmax, ref_elements, atomic_numbers)
        k_NM_flat = kernel_nm_flatten(kernel_size, kernel_sparse_indices, ref_elements, atomic_numbers, k_NM)
        np.savetxt(f'{kernel_file}.dat', k_NM_flat)


def kernel_mm(lmax, power_ref):

    samples = {}
    k_MM = {}
    for (l, q), rblock in power_ref.items():
        msize = 2*l+1
        nsamp = len(rblock.samples)
        if q not in samples:
            samples[q] = list(rblock.samples)
        k_MM[(l, q)] = np.zeros((nsamp, nsamp, msize, msize))
        for iiref1 in range(nsamp):
            vec1 = rblock.values[iiref1]
            for iiref2 in range(iiref1, nsamp):
                vec2 = rblock.values[iiref2]
                dot = vec1 @ vec2.T
                k_MM[(l, q)][iiref1, iiref2] = dot
                if iiref1!=iiref2:
                    k_MM[(l, q)][iiref2, iiref1] = dot.T
    for lm, q in lmax.items():
        # Mind the descending order of l
        for l in range(lm, -1, -1):
            k_MM[(l, q)] *= k_MM[(0, q)]

    return kmm2tmap(samples, k_MM)
