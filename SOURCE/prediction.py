#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, get_training_set
from libs.basis import basis_read
from libs.predict import run_prediction
from libs.tmap import join


def main():
    o, p = read_config(sys.argv)
    training = 'training' in sys.argv[1:]

    atomic_numbers = moldata_read(p.xyzfilename)
    lmax, nmax = basis_read(p.basisfilename)
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]

    for frac in o.fracs:
        weights = np.load(f'{p.weightsfilebase}_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy')
        if not training:
            ntest, test_configs = get_test_set(p.trainfilename, len(atomic_numbers))
            predictfile = f'{p.predictfilebase}_test_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npz'
        else:
            ntest, test_configs = get_training_set(p.trainfilename, frac)
            predictfile = f'{p.predictfilebase}_training_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npz'

        print(f'Number of testing molecules = {ntest}')
        predictions = run_prediction(test_configs, atomic_numbers[test_configs],
                                     lmax, nmax, weights, ref_elements,
                                     p.kernelconfbase)
        predictions = join(predictions)
        equistore.save(predictfile, predictions)


if __name__=='__main__':
    main()
