#!/usr/bin/env python3
"""Preprocess the dataset."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import ase.io
import metatensor
from qstack import reorder
from qstack.io import metatensor as equio
from libs.config import get_settings
from libs.functions import get_elements, Basis, make_dummy_mol
from libs.tmap import averages2tmap
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()
    logger.info(f'{o.process_metric=}')

    mol_names, atomic_numbers = prepare_molecules(p)

    nenv = get_elements(atomic_numbers, return_counts=True)

    basis = Basis(o.basisname, elements=nenv.keys())
    ao_indices = [basis.index(atoms) for atoms in atomic_numbers]

    coefficients = load_coefs(mol_names, p.input_coeffs)

    av_coefs = get_averages(nenv, basis, coefficients, ao_indices)

    for imol, (mol_name, coef, atoms, ao_index) in tqdm([*enumerate(zip(mol_names, coefficients, atomic_numbers, ao_indices, strict=True))]):

        mol = make_dummy_mol(atoms, basis=o.basisname, ignore=True)

        coef = reorder.reorder_ao(mol, coef, dest='gpr', src=o.coeff_order)
        np.save(p.clean_coefficients.format(imol), coef)
        coef = remove_averages(ao_index, coef, av_coefs)

        if o.process_metric:
            over = np.load(p.input_metrics.format(mol_name=mol_name))
            over = reorder.reorder_ao(mol, over, dest='gpr', src=o.overlap_order)
            metatensor.save(p.metric_matrix.format(imol), equio.array_to_tensormap(mol, over, src='gpr'))
        else:
            over = equio.tensormap_to_array(mol, metatensor.load(p.metric_matrix.format(imol)), dest='gpr', fast=True)

        proj = over @ coef
        metatensor.save(p.projection.format(imol), equio.array_to_tensormap(mol, proj, src='gpr'))
    metatensor.save(p.spherical_averages, averages2tmap(av_coefs))


def prepare_molecules(p):
    """Read dataset molecules, write a combined XYZ file, and return IDs with atomic numbers.

    Args:
        p (types.SimpleNamespace): Paths namespace from get_settings().

    Returns:
        tuple[list[str], list[np.ndarray]]: Molecule IDs and per-molecule atomic-number arrays.
    """
    df = pd.read_csv(p.dataset)
    mol_names = df['id'].to_list()
    asemols = [ase.io.read(p.xyz.format(mol_name=mol_name)) for mol_name in mol_names]
    ase.io.write(p.xyzfilename, asemols)
    atomic_numbers = [asemol.numbers for asemol in asemols]
    return mol_names, atomic_numbers


def load_coefs(mol_names, fname_template):
    """Load coefficient vectors for all molecules from a filename template.

    Args:
        mol_names (list[str]): Molecule IDs used to format template paths.
        fname_template (str): Template path to coefficient files (.npy or text).

    Returns:
        list[np.ndarray]: Loaded coefficient vectors in molecule order.
    """
    return [np.load(fname_template.format(mol_name=mol_name)) if (fname := fname_template.format(mol_name=mol_name)).endswith('.npy') else np.loadtxt(fname) for mol_name in mol_names]


def get_averages(nenv, basis, coefficients, ao_indices):
    """Compute per-element average l=0 coefficients across molecules.

    Args:
        nenv (dict[int, int]): Number of environments per element.
        basis (Basis): Basis.
        coefficients (list[np.ndarray]): Per-molecule coefficient vectors.
        ao_indices (list[.functions.AOIndex]): Per-molecule AO indices.

    Returns:
        dict[int, np.ndarray]: Average l=0 coefficient vector for each element.
    """
    av_coefs = {q: np.zeros(basis.nmax[q][0]) for q in nenv}

    for coef, ao_index in zip(coefficients, ao_indices, strict=True):
        for iat, q in enumerate(ao_index.atoms):
            av_coefs[q] += coef[ao_index.find(iat=iat, l=0)]

    for q in av_coefs:
        av_coefs[q] /= nenv[q]
    return av_coefs


def remove_averages(ao_index, coef, av_coefs):
    """Subtract per-element l=0 averages from one coefficient vector.

    Args:
        ao_index (AOIndex): AO index.
        coef (np.ndarray): Input coefficient vector.
        av_coefs (dict[int, np.ndarray]): Per-element average l=0 coefficients.

    Returns:
        np.ndarray: Coefficient vector with averages removed.
    """
    coef_new = np.copy(coef)
    for iat, q in enumerate(ao_index.atoms):
        coef_new[ao_index.find(iat=iat, l=0)] -= av_coefs[q]
    return coef_new


if __name__=='__main__':
    main()
