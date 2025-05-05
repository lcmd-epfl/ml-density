#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from libs.config import read_config
from libs.basis import basis_read
from libs.functions import moldata_read, get_elements_list, print_progress
from libs.kernels_lib import kernel_for_mol


def main():
    o, p = read_config(sys.argv)

    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    lmax, _ = basis_read(p.basisfilename)
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]
    power_ref = metatensor.load(f'{p.powerrefbase}_{o.M}.mts');

    for imol, atoms in enumerate(atomic_numbers_ex):
        print_progress(imol, len(atomic_numbers_ex))
        kernel_for_mol(lmax, ref_elements, atoms, power_ref,
                       f'{p.powerexbase}_{imol}.mts',
                       f'{p.kernelexbase}{imol}.dat')


if __name__=='__main__':
    main()
