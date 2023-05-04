#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import read_config
from basis import basis_read
from functions import moldata_read, get_elements_list, get_atomicindx, print_progress
from libs.kernels_lib import kernel_for_mol
from libs.multi import multi_process

USEMPI = 1


def main():
    o, p = read_config(sys.argv)

    def do_mol(imol):
        kernel_for_mol(lmax, ref_elements, atomic_numbers[imol],
                       power_ref, f'{p.splitpsfilebase}_{imol}.npz', f'{p.kernelconfbase}{imol}.dat')

    (nmol, _, atomic_numbers) = moldata_read(p.xyzfilename)
    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    ref_elements = np.hstack(atomic_numbers)[ref_indices]

    power_ref = equistore.load(f'{p.powerrefbase}_{o.M}.npz');

    elements = get_elements_list(atomic_numbers)
    (el_dict, lmax, nmax) = basis_read(p.basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    if USEMPI==0:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        multi_process(nmol, do_mol)


if __name__=='__main__':
    main()
