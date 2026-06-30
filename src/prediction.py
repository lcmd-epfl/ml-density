#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, get_training_set, Basis
from libs.predict import run_prediction
from libs.tmap import join


def main():
    o, p = read_config(sys.argv)
    training = 'training' in sys.argv[1:]

    atomic_numbers = moldata_read(p.xyzfilename)
    ref_elements = pd.read_csv(f'{p.refsselfilebase}{o.M}.csv')['q'].to_numpy()
    basis = Basis(o.basisname, elements=np.unique(ref_elements))

    for frac in o.fracs:
        weights = np.load(f'{p.weightsfilebase}_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy')
        if not training:
            ntest, test_configs = get_test_set(p.trainfilename, len(atomic_numbers))
            predictfile = f'{p.predictfilebase}_test_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.mts'
        else:
            ntest, test_configs = get_training_set(p.trainfilename, frac)
            predictfile = f'{p.predictfilebase}_training_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.mts'

        print(f'Number of testing molecules = {ntest}')
        predictions = run_prediction(test_configs, atomic_numbers[test_configs],
                                     basis, weights, ref_elements,
                                     p.kernelconfbase)
        predictions = join(predictions)
        metatensor.save(predictfile, predictions)


if __name__=='__main__':
    main()
