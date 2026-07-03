#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from tqdm import tqdm
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, Basis
from libs.tmap import split, tmap2vector, tmap_add


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzfilename)
    averages = metatensor.load(p.spherical_averages)

    basis = Basis(o.basisname, elements=set(averages.keys.column('center_type')))
    _, test_configs = get_test_set(p.train_test_sets)

    for frac in o.fracs:
        print('fraction =', frac)
        predictfile = p.predictions.format(subset='test', train_frac=frac)
        predictions = split(metatensor.load(predictfile))
        for imol, c in zip(tqdm(test_configs), predictions, strict=True):
            tmap_add(c, averages)
            rho = tmap2vector(atomic_numbers[imol], basis, c)
            np.savetxt(f'{p.outfilebase}tf{frac}_gpr_{imol}.dat', rho)


if __name__=='__main__':
    main()
