#!/usr/bin/env python3

import sys, os
import numpy as np
import equistore
from libs.config import read_config
from libs.basis import basis_read
from libs.functions import moldata_read, print_progress
from libs.kernels_lib import kernel_for_mol
from libs.multi import multi_process

USEMPI = 1


def main():
    o, p = read_config(sys.argv)

    def do_mol(imol):
        #if os.path.exists(f'{p.kernelconfbase}{imol}.dat'):
        #    return
        kernel_for_mol(lmax, ref_elements, atomic_numbers[imol],
                       power_ref, f'{p.splitpsfilebase}_{imol}.npz', f'{p.kernelconfbase}{imol}.npz')

    atomic_numbers = moldata_read(p.xyzfilename)
    power_ref = equistore.load(f'{p.powerrefbase}_{o.M}.npz');
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]
    lmax, nmax = basis_read(p.basisfilename)
    nmol = len(atomic_numbers)

    if USEMPI==0:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        multi_process(nmol, do_mol)


if __name__=='__main__':
    main()
