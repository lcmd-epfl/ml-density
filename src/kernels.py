#!/usr/bin/env python3

import sys
import pandas as pd
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, print_progress, Basis
from libs.kernels_lib import kernel_for_mol
from libs.multi import multi_process

USEMPI = 1


def main():
    o, p = read_config(sys.argv)

    def do_mol(imol):
        #import os
        #if os.path.exists(f'{p.kernelconfbase}{imol}.dat'):
        #    return
        kernel_for_mol(basis.lmax, ref_elements, atomic_numbers[imol],
                       power_ref, p.power_spectrum.format(imol), f'{p.kernelconfbase}{imol}.mts')

    atomic_numbers = moldata_read(p.xyzfilename)
    power_ref = metatensor.load(p.reference_power_spectra)
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=set(ref_elements))
    nmol = len(atomic_numbers)

    if USEMPI==0:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        multi_process(nmol, do_mol)


if __name__=='__main__':
    main()
