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

    _, _, atomic_numbers = moldata_read(p.xyzfilename)
    nmol_ex, _, atomic_numbers_ex = moldata_read(p.xyzexfilename)

    elements = get_elements_list(atomic_numbers)

    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    ref_elements = np.hstack(atomic_numbers)[ref_indices]

    # elements dictionary, max. angular momenta, number of radial channels
    (el_dict, lmax, _) = basis_read(p.basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    power_ref = equistore.load(f'{p.powerrefbase}_{o.M}.npz');

    for imol in range(nmol_ex):
        print_progress(imol, nmol_ex)
        kernel_for_mol(lmax, ref_elements, atomic_numbers_ex[imol],
                       power_ref, f'{p.powerexbase}_{imol}.npz', f'{p.kernelexbase}{imol}.dat')


if __name__=='__main__':
    main()
