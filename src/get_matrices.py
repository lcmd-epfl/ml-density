#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from libs.config import read_config
from libs.functions import get_training_sets, Basis
from libs.get_matrices_A import get_a
from libs.get_matrices_B import get_b


def main():
    o, p = read_config(sys.argv)

    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=set(ref_elements))

    # training set selection
    ntrains, train_configs = get_training_sets(p.train_test_sets, o.fracs)
    ntrains = np.pad(ntrains, (0, 1), 'constant', constant_values=0)

    if len(sys.argv)>1 and sys.argv[1][0].lower()=='b':
        bmatfiles = [f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat' for frac in o.fracs]
        get_b(basis, ref_elements, ntrains, train_configs,
              p.goodoverfilebase, p.kernelconfbase, bmatfiles)
    else:
        avecfiles = [f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt' for frac in o.fracs]
        get_a(basis, ref_elements, ntrains, train_configs,
              p.baselinedwbase, p.kernelconfbase, avecfiles)


if __name__=='__main__':
    main()
