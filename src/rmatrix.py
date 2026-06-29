#!/usr/bin/env python3

import sys
import metatensor
from libs.functions import Basis
from libs.config import read_config
from libs.kernels_lib import kernel_mm


def main():
    o, p = read_config(sys.argv)

    power_ref = metatensor.load(f'{p.powerrefbase}_{o.M}.mts')
    basis = Basis(o.basisname, elements=set(power_ref.keys.column('center_type')))

    k_MM = kernel_mm(basis.lmax, power_ref)
    metatensor.save(f'{p.kmmbase}{o.M}.mts', k_MM)


if __name__=='__main__':
    main()
