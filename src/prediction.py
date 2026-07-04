#!/usr/bin/env python3

import sys
import pandas as pd
import metatensor
from qstack.io.metatensor import join
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, get_training_set, Basis
from libs.predict import run_prediction


def main():
    args, o, p = read_config(sys.argv, return_args=['training'])

    atomic_numbers = moldata_read(p.xyzfilename)
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=ref_elements)

    for frac in o.fracs:
        weights = metatensor.load(p.weights.format(train_frac=frac))
        if args.training:
            ntest, test_configs = get_training_set(p.train_test_sets, frac)
            predictfile = p.predictions.format(subset='training', train_frac=frac)
        else:
            ntest, test_configs = get_test_set(p.train_test_sets)
            predictfile = p.predictions.format(subset='test', train_frac=frac)

        print(f'Number of testing molecules = {ntest}')
        predictions = run_prediction(test_configs, atomic_numbers[test_configs],
                                     basis, weights, p.kernel_nm)
        predictions = join(predictions)
        metatensor.save(predictfile, predictions)


if __name__=='__main__':
    main()
