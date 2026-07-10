#!/usr/bin/env python3
"""Compute λ-SOAP power spectra for the dataset molecules."""

import os
import numpy as np
import ase.io
from libs.config import get_settings
from libs.functions import get_elements, moldata_read, Basis, get_dataset_paths
from libs.logger_setup import setup_logger
from libs.config_utils import defaults
from libs.lsoap import generate_lambda_soap_wrapper, make_rascal_hypers
from libs.multi import process_molecules
import metatensor

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['missing_only', 'mpi', 'extra'])
    dataset_paths = get_dataset_paths(p, args.extra)
    run_power_spectrum(dataset_paths, p.xyzfilename, o, missing_only=args.missing_only, use_mpi=args.mpi)


def run_power_spectrum(dataset_paths, xyz_path_main, o, *, missing_only=False, use_mpi=defaults.mpi):
    """Compute and save per-molecule power spectra.

    Args:
        dataset_paths (.libs.functions.DatasetPaths): Paths and path templates
                      (XYZ, power spectra, kernel) for the main or out-of-sample dataset.
        xyz_path_main (str): Path to the *main* dataset XYZ file.
        o (SimpleNamespace): Options parsed from config.
        missing_only (bool): Skip molecules with existing output files.
        use_mpi (bool): Whether to use MPI for process distribution.
    """
    def do_mol(imol):
        ppath = dataset_paths.power.format(imol)
        if missing_only and os.path.exists(ppath):
            return
        soap = generate_lambda_soap_wrapper(mols[imol], rascal_hypers, neighbor_species=elements,
                                            normalize=o.ps_normalize, min_norm=o.ps_min_norm, lmax=basis.lmax)
        metatensor.save(ppath, soap)

    mols, elements = read_mol_elements(dataset_paths.xyz, xyz_path_main)

    basis = Basis(o.basisname, elements)
    rascal_hypers = make_rascal_hypers(o.soap_rcut, o.soap_ncut, o.soap_lcut, o.soap_sigma)
    logger.debug(f'rascal_hypers={rascal_hypers}')
    logger.debug(f'elements={elements}')
    logger.debug(f'basis.lmax={basis.lmax}')
    logger.debug(f'ps_min_norm={o.ps_min_norm} ps_normalize={o.ps_normalize}')
    process_molecules(len(mols), do_mol, use_mpi=use_mpi)


def read_mol_elements(xyz_this_set, xyz_main_set):
    """Check if out-of-sample elements are present in the dataset.

    Args:
        xyz_this_set (str): Path to the XYZ file (main or out-of-sample dataset).
        xyz_main_set (str): Path to the main dataset XYZ file.

    Returns:
        np.ndarray[int]: Elements present in the *main* dataset.

    Raises:
        RuntimeError: Out-of-sample set contains elements that are not in the main dataset.
    """
    mols = ase.io.read(xyz_this_set, ":")
    q_this_set = get_elements(mols)

    if xyz_this_set==xyz_main_set:
        q = q_this_set
    else:
        q_main_set = get_elements(moldata_read(xyz_main_set))
        if not set(q_this_set).issubset(q_main_set):
            with np.printoptions(legacy="1.25"):
                msg = f'Out-of-sample molecules contain elements not present in the dataset: {set(q_this_set).difference(q_main_set)}'
            raise RuntimeError(msg)
        logger.debug(f'OSS contains: {q_this_set}')
        q = q_main_set
    return mols, q


if __name__=='__main__':
    main()
