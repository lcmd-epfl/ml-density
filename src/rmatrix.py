#!/usr/bin/env python3

import sys
import metatensor
from libs.functions import Basis
from libs.config import read_config
from libs.kernels_lib import kernel_mm


def main():
    o, p = read_config(sys.argv)

    power_ref = metatensor.load(p.reference_power_spectra)
    basis = Basis(o.basisname, elements=power_ref.keys.column('center_type'))

    k_MM = kernel_mm(basis.lmax, power_ref)
    metatensor.save(p.kernel_mm, k_MM)


if __name__=='__main__':
    main()
