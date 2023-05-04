#!/usr/bin/env python3

import sys
import numpy as np
from basis import basis_read
from config import read_config
from functions import moldata_read, get_elements_list
from libs.kernels_lib import kernel_mm


def main():
    o, p = read_config(sys.argv)

    _, _, atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)

    (el_dict, _, _) = basis_read(p.basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    ref_elements = np.hstack(atomic_numbers)[ref_indices]
    k_MM = kernel_mm(o.M, p.powerrefbase, ref_elements)
    np.save(f'{p.kmmbase}{o.M}.npy', k_MM )


if __name__=='__main__':
    main()
