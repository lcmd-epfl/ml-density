#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import read_config
from basis import basis_read
from functions import moldata_read, get_elements_list, print_progress
from libs.kernels_lib import kernel_for_mol


def main():
    o, p = read_config(sys.argv)

    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    lmax, _ = basis_read(p.basisfilename)
    ref_elements = np.loadtxt(f'{p.qrefsselfilebase}{o.M}.txt', dtype=int)
    power_ref = equistore.load(f'{p.powerrefbase}_{o.M}.npz');

    for imol, atoms in enumerate(atomic_numbers_ex):
        print_progress(imol, len(atomic_numbers_ex))
        kernel_for_mol(lmax, ref_elements, atoms, power_ref,
                       f'{p.powerexbase}_{imol}.npz',
                       f'{p.kernelexbase}{imol}.dat')


if __name__=='__main__':
    main()
