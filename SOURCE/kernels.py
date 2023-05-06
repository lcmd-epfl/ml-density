#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import read_config
from basis import basis_read
from functions import moldata_read, print_progress
from libs.kernels_lib import kernel_for_mol
from libs.multi import multi_process
import os

USEMPI = 1


def main():
    o, p = read_config(sys.argv)

    def do_mol(imol):
        #if os.path.exists(f'{p.kernelconfbase}{imol}.dat'):
        #    return
        kernel_for_mol(lmax, ref_elements, atomic_numbers[imol],
                       power_ref, f'{p.splitpsfilebase}_{imol}.npz', f'{p.kernelconfbase}{imol}.dat')

    nmol, _, atomic_numbers = moldata_read(p.xyzfilename)
    power_ref = equistore.load(f'{p.powerrefbase}_{o.M}.npz');
    ref_elements = np.loadtxt(f'{p.qrefsselfilebase}{o.M}.txt', dtype=int)
    lmax, nmax = basis_read(p.basisfilename)

    if USEMPI==0:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        multi_process(nmol, do_mol)


if __name__=='__main__':
    main()
