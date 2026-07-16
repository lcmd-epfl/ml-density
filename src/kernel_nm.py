#!/usr/bin/env python3
"""Compute per-molecule kernels against reference environments."""

import os
import metatensor
from libs.config import get_settings
from libs.logger_setup import setup_logger
from libs.functions import moldata_read, get_dataset_paths
from libs.kernels_lib import kernel_for_mol
from libs.multi import process_molecules
from libs.config_utils import defaults

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    args, _, p = get_settings(return_args=['missing_only', 'mpi', 'extra'])
    dataset_paths = get_dataset_paths(p, extra=args.extra)
    run_kernel(dataset_paths, p.reference_power_spectra, missing_only=args.missing_only, use_mpi=args.mpi)


def run_kernel(dataset_paths, ref_power_path, *, missing_only=False, use_mpi=defaults.mpi):
    """Compute and save per-molecule kernels.

    Args:
        dataset_paths (.libs.functions.DatasetPaths): Paths and path templates
                      (XYZ, power spectra, kernel) for the main or out-of-sample dataset.
        ref_power_path (str): Path to the reference power spectra.
        missing_only (bool): Skip molecules with existing kernel files.
        use_mpi (bool): Whether to use MPI for process distribution.
    """
    def do_mol(imol):
        kpath = dataset_paths.kernel.format(imol)
        if missing_only and os.path.exists(kpath):
            return
        kernel_for_mol(atomic_numbers[imol], power_ref, dataset_paths.power.format(imol), kpath)

    atomic_numbers = moldata_read(dataset_paths.xyz)
    power_ref = metatensor.load(ref_power_path)
    process_molecules(len(atomic_numbers), do_mol, use_mpi=use_mpi)


if __name__=='__main__':
    main()
