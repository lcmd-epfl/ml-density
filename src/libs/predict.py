"""Predict AO coefficients."""

import numpy as np
import metatensor
from libs.progress import tqdm
from libs.tmap import vector2tmap


def compute_prediction(atoms, basis, kernel, weights, averages=None):
    """Predict AO coefficients for one molecule from kernels and weights.

    Args:
        atoms (np.ndarray | list[int]): Atomic numbers of the target molecule.
        basis (.functions.Basis): Basis used for AO indexing.
        kernel (metatensor.TensorMap): Kernel between the target molecule and reference environments.
        weights (metatensor.TensorMap): Regression weights.
        averages (metatensor.TensorMap | None): Optional l=0 averages added back to predictions.

    Returns:
        metatensor.TensorMap: Predicted coefficient TensorMap for the molecule.
    """
    nao = basis.nao_for_mol(atoms)
    coeffs = vector2tmap(atoms, basis.llist, np.zeros(nao))
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


def run_prediction(test_configs, atomic_numbers,
                   basis, weights, path_kern, averages=None):
    """Run predictions for multiple molecules and return TensorMap outputs.

    Args:
        test_configs (list[int] | np.ndarray): Molecule indices to predict.
        atomic_numbers (list[np.ndarray]): Atomic numbers for each selected molecule.
        basis (Basis): Basis helper used for AO indexing.
        weights (metatensor.TensorMap): Regression weights TensorMap.
        path_kern (str): Template path to per-molecule kernel files.
        averages (metatensor.TensorMap | None): Optional l=0 averages added back to predictions.

    Returns:
        list[metatensor.TensorMap]: Predicted coefficients for all requested molecules.
    """
    predictions = []
    for imol, atoms in zip(tqdm(test_configs), atomic_numbers, strict=True):
        kernel = metatensor.load(path_kern.format(imol))
        predictions.append(compute_prediction(atoms, basis, kernel, weights, averages=averages))
    return predictions
