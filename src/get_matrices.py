#!/usr/bin/env python3

import sys
import numpy as np
from libs.config import read_config
from libs.functions import get_training_sets, Basis
from libs.get_matrices_A import get_a
from libs.get_matrices_B import get_b


def main():
    o, p = read_config(sys.argv)

    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]
    basis = Basis(o.basisname, elements=set(ref_elements))

    # training set selection
    nfrac, ntrains, train_configs = get_training_sets(p.trainfilename, o.fracs)

    if len(sys.argv)>1 and sys.argv[1][0].lower()=='b':
        bmatfiles = [f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat' for frac in o.fracs]
        get_b(basis, ref_elements, nfrac, ntrains, train_configs,
              p.goodoverfilebase, p.kernelconfbase, bmatfiles)
    else:
        avecfiles = [f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt' for frac in o.fracs]
        get_a(basis, ref_elements, nfrac, ntrains, train_configs,
              p.baselinedwbase, p.kernelconfbase, avecfiles)


if __name__=='__main__':
    main()
