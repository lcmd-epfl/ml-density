#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from libs.basis import basis_read
from libs.config import read_config
from libs.kernels_lib import kernel_mm


def main():
    o, p = read_config(sys.argv)

    lmax, _ = basis_read(p.basisfilename)
    power_ref = equistore.load(f'{p.powerrefbase}_{o.M}.npz')
    k_MM = kernel_mm(lmax, power_ref)
    equistore.save(f'{p.kmmbase}{o.M}.npz', k_MM)


if __name__=='__main__':
    main()
