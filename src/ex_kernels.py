#!/usr/bin/env python3

import sys
import metatensor
from tqdm import tqdm
from libs.config import read_config
from libs.functions import moldata_read
from libs.kernels_lib import kernel_for_mol


def main():
    _, p = read_config(sys.argv)

    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    power_ref = metatensor.load(p.reference_power_spectra)

    for imol, atoms in enumerate(tqdm(atomic_numbers_ex)):
        kernel_for_mol(atoms, power_ref, p.extra_power_spectrum.format(imol), p.extra_kernel_nm.format(imol))


if __name__=='__main__':
    main()
