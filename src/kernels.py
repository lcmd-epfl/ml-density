#!/usr/bin/env python3

import os
import metatensor
from tqdm import trange
from libs.config import get_settings
from libs.functions import moldata_read
from libs.kernels_lib import kernel_for_mol
from libs.multi import multi_process
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():
    args, _, p = get_settings(return_args=['missing_only', 'mpi'])

    def do_mol(imol):
        kpath = p.kernel_nm.format(imol)
        if args.missing_only and os.path.exists(kpath):
            return
        kernel_for_mol(atomic_numbers[imol], power_ref, p.power_spectrum.format(imol), kpath)

    atomic_numbers = moldata_read(p.xyzfilename)
    power_ref = metatensor.load(p.reference_power_spectra)
    nmol = len(atomic_numbers)

    if args.mpi:
        multi_process(nmol, do_mol)
    else:
        for imol in trange(nmol):
            do_mol(imol)


if __name__=='__main__':
    main()
