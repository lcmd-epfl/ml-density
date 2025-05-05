import numpy as np
import equistore
from libs.functions import nao_for_mol, print_progress
from libs.tmap import vector2tmap


def compute_prediction(atoms, lmax, nmax, kernel, weights, averages=None):
    nao = nao_for_mol(atoms, lmax, nmax)
    coeffs = vector2tmap(atoms, lmax, nmax, np.zeros(nao))
    for (l, q), cblock in coeffs:
        wblock = weights.block(spherical_harmonics_l=l, species_center=q)
        kblock = kernel.block(spherical_harmonics_l=l, species_center=q)
        for sample in cblock.samples:
            cpos = cblock.samples.position(sample)
            kpos = kblock.samples.position(sample)
            cblock.values[cpos,:,:] = np.einsum('mMr,rMn->mn', kblock.values[kpos], wblock.values)
        if averages and l==0:
            cblock.values[:,:,:] = cblock.values + averages.block(species_center=q).values
    return coeffs


def run_prediction(test_configs, atomic_numbers,
                   lmax, nmax, weights, ref_elements,
                   kernelbase, averages=None):

    weights = vector2tmap(ref_elements, lmax, nmax, weights)

    predictions = []
    for i, (imol, atoms) in enumerate(zip(test_configs, atomic_numbers)):
        print_progress(i, len(test_configs))
        kernel = equistore.load(f'{kernelbase}{imol}.npz')
        predictions.append(compute_prediction(atoms, lmax, nmax, kernel, weights, averages=averages))
    return predictions
