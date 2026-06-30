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
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=np.unique(ref_elements))

    for frac in o.fracs:
        weights = np.load(p.weights.format(train_frac=frac))
        if not training:
            ntest, test_configs = get_test_set(p.trainfilename, len(atomic_numbers))
            predictfile = p.predictions.format(subset='test', train_frac=frac)
        else:
            ntest, test_configs = get_training_set(p.trainfilename, frac)
            predictfile = p.predictions.format(subset='training', train_frac=frac)

        print(f'Number of testing molecules = {ntest}')
        predictions = run_prediction(test_configs, atomic_numbers[test_configs],
                                     basis, weights, ref_elements,
                                     p.kernelconfbase)
        predictions = join(predictions)
        metatensor.save(predictfile, predictions)


if __name__=='__main__':
    main()
