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
