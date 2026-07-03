#!/usr/bin/env python3

import sys
import pandas as pd
import metatensor
from tqdm import tqdm
from libs.config import read_config
from libs.functions import moldata_read, Basis
from libs.kernels_lib import kernel_for_mol


def main():
    o, p = read_config(sys.argv)

    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    power_ref = metatensor.load(p.reference_power_spectra)
    basis = Basis(o.basisname, elements=ref_elements)

    for imol, atoms in enumerate(tqdm(atomic_numbers_ex)):
        kernel_for_mol(basis.lmax, ref_elements, atoms, power_ref,
                       p.extra_power_spectrum.format(imol),
                       p.extra_kernel_nm.format(imol))


if __name__=='__main__':
    main()
