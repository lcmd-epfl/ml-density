#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, print_progress, Basis
from libs.kernels_lib import kernel_for_mol


def main():
    o, p = read_config(sys.argv)

    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]
    power_ref = metatensor.load(f'{p.powerrefbase}_{o.M}.mts')
    basis = Basis(o.basisname, elements=set(ref_elements))

    for imol, atoms in enumerate(atomic_numbers_ex):
        print_progress(imol, len(atomic_numbers_ex))
        kernel_for_mol(basis.lmax, ref_elements, atoms, power_ref,
                       f'{p.powerexbase}_{imol}.mts',
                       f'{p.kernelexbase}{imol}.mts')


if __name__=='__main__':
    main()
