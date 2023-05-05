#!/usr/bin/env python3

import sys
import numpy as np
from config import read_config
from basis import basis_read
from functions import moldata_read, get_test_set, get_training_set
from libs.predict import run_prediction
from libs.tmap import join
import equistore


def main():
    o, p = read_config(sys.argv)
    training = 'training' in sys.argv[1:]

    nmol, _, atomic_numbers = moldata_read(p.xyzfilename)
    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    ref_elements= np.hstack(atomic_numbers)[ref_indices]

    for frac in o.fracs:
        if not training:
            ntest, test_configs = get_test_set(p.trainfilename, nmol)
            predictfile = f'{p.predictfilebase}_test_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npz'
        else:
            ntest, test_configs = get_training_set(p.trainfilename, frac)
            predictfile = f'{p.predictfilebase}_training_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npz'
        weightsfile = f'{p.weightsfilebase}_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'

        print(f'Number of testing molecules = {ntest}')
        predictions = run_prediction(test_configs, atomic_numbers[test_configs], ref_elements,
                                     p.basisfilename, weightsfile, p.kernelconfbase)
        predictions = join(predictions)
        equistore.save(predictfile, predictions)


if __name__=='__main__':
    main()
