#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from libs.config import read_config
from libs.basis import basis_read
from libs.functions import moldata_read, get_test_set, print_progress
from libs.tmap import split, tmap2vector, tmap_add


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzfilename)
    lmax, nmax = basis_read(p.basisfilename)
    ntest, test_configs = get_test_set(p.trainfilename, len(atomic_numbers))
    averages = metatensor.load(p.avfile)

    for frac in o.fracs:
        print('fraction =', frac)
        predictfile = f'{p.predictfilebase}_test_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.mts'
        predictions = split(metatensor.load(predictfile))
        for itest, (imol, c) in enumerate(zip(test_configs, predictions)):
            print_progress(itest, ntest)
            tmap_add(c, averages)
            rho = tmap2vector(atomic_numbers[imol], lmax, nmax, c)
            np.savetxt(f'{p.outfilebase}tf{frac}_gpr_{imol}.dat', rho)


if __name__=='__main__':
    main()
