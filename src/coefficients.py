#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, print_progress, Basis
from libs.tmap import split, tmap2vector, tmap_add


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzfilename)
    averages = metatensor.load(p.avfile)

    basis = Basis(o.basisname, elements=set(averages.keys.column('center_type')))
    ntest, test_configs = get_test_set(p.trainfilename, len(atomic_numbers))

    for frac in o.fracs:
        print('fraction =', frac)
        predictfile = p.predictions.format(subset='test', train_frac=frac)
        predictions = split(metatensor.load(predictfile))
        for itest, (imol, c) in enumerate(zip(test_configs, predictions, strict=True)):
            print_progress(itest, ntest)
            tmap_add(c, averages)
            rho = tmap2vector(atomic_numbers[imol], basis, c)
            np.savetxt(f'{p.outfilebase}tf{frac}_gpr_{imol}.dat', rho)


if __name__=='__main__':
    main()
