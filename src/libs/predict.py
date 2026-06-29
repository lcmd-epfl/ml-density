import numpy as np
import metatensor
from libs.functions import nao_for_mol, print_progress
from libs.tmap import vector2tmap, sparseindices_fill


def compute_prediction(atoms: np.ndarray, lmax: dict, nmax: dict, kernel: metatensor.TensorMap, weights: metatensor.TensorMap, averages: metatensor.TensorMap = None) -> metatensor.TensorMap:
    '''Predict density coefficients for one molecule from kernel and regression weights.'''
    nao = nao_for_mol(atoms, lmax, nmax)
    coeffs = vector2tmap(atoms, lmax, nmax, np.zeros(nao))
    for (l, q), cblock in coeffs.items():
        wblock = weights.block(o3_lambda=l, center_type=q)
        kblock = kernel.block(o3_lambda=l, center_type=q)
        for sample in cblock.samples:
            cpos = cblock.samples.position(sample)
            kpos = kblock.samples.position(sample)
            cblock.values[cpos,:,:] = np.einsum('mMr,rMn->mn', kblock.values[kpos], wblock.values)
        if averages and l==0:
            cblock.values[:,:,:] = cblock.values + averages.block(center_type=q).values
    return coeffs


def compute_prediction_variance(atoms: np.ndarray, lmax: dict, nmax: dict, kernel: metatensor.TensorMap, weight_cov: np.ndarray, ref_elements: np.ndarray) -> metatensor.TensorMap:
    '''Compute per-coefficient prediction variance for one molecule.'''
    nao = nao_for_mol(atoms, lmax, nmax)
    variances = vector2tmap(atoms, lmax, nmax, np.zeros(nao))
    idx = sparseindices_fill(lmax, nmax, ref_elements)
    ref_by_q = {q: np.where(ref_elements == q)[0] for q in np.unique(ref_elements)}

    for (l, q), vblock in variances.items():
        if q not in ref_by_q:
            continue

        ref_idx = ref_by_q[q]
        nref = len(ref_idx)
        msize = 2 * l + 1
        kblock = kernel.block(o3_lambda=l, center_type=q)

        cov_by_n = []
        for n in range(nmax[(q, l)]):
            inds = np.empty(nref * msize, dtype=int)
            for i, iref in enumerate(ref_idx):
                base = idx[iref, l] + n * msize
                inds[i * msize:(i + 1) * msize] = np.arange(base, base + msize)
            cov_by_n.append(weight_cov[np.ix_(inds, inds)])

        for sample in vblock.samples:
            vpos = vblock.samples.position(sample)
            kpos = kblock.samples.position(sample)
            kvals = kblock.values[kpos]

            kvecs = []
            for im in range(msize):
                kvec = np.empty(nref * msize)
                for iref_local in range(nref):
                    sl = slice(iref_local * msize, (iref_local + 1) * msize)
                    kvec[sl] = kvals[im, :, iref_local]
                kvecs.append(kvec)

            for n, cov in enumerate(cov_by_n):
                for im, kvec in enumerate(kvecs):
                    vblock.values[vpos, im, n] = np.einsum('a,ab,b->', kvec, cov, kvec, optimize=True)

    return variances


def run_prediction(test_configs: np.ndarray, atomic_numbers: np.ndarray,
                   lmax: dict, nmax: dict, weights: np.ndarray, ref_elements: np.ndarray,
                   kernelbase: str, averages: metatensor.TensorMap = None) -> list[metatensor.TensorMap]:

    weights = vector2tmap(ref_elements, lmax, nmax, weights)

    predictions = []
    for i, (imol, atoms) in enumerate(zip(test_configs, atomic_numbers)):
        print_progress(i, len(test_configs))
        kernel = metatensor.load(f'{kernelbase}{imol}.mts')
        predictions.append(compute_prediction(atoms, lmax, nmax, kernel, weights, averages=averages))
    return predictions


def run_prediction_with_variance(test_configs: np.ndarray, atomic_numbers: np.ndarray,
                                 lmax: dict, nmax: dict, weights: np.ndarray, weight_cov: np.ndarray, ref_elements: np.ndarray,
                                 kernelbase: str, averages: metatensor.TensorMap = None) -> tuple[list[metatensor.TensorMap], list[metatensor.TensorMap]]:

    weights = vector2tmap(ref_elements, lmax, nmax, weights)

    predictions = []
    variances = []
    for i, (imol, atoms) in enumerate(zip(test_configs, atomic_numbers)):
        print_progress(i, len(test_configs))
        kernel = metatensor.load(f'{kernelbase}{imol}.mts')
        predictions.append(compute_prediction(atoms, lmax, nmax, kernel, weights, averages=averages))
        variances.append(compute_prediction_variance(atoms, lmax, nmax, kernel, weight_cov, ref_elements))
    return predictions, variances
