#!/usr/bin/env python3

import sys
import numpy as np
from libs.config import read_config
from libs.basis import basis_read
from libs.functions import nao_for_mol, get_training_sets
from libs.get_matrices_A import get_a
from libs.get_matrices_B import get_b


def main():
    o, p = read_config(sys.argv)

    lmax, nmax = basis_read(p.basisfilename)

    # reference environments
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]
    totsize = nao_for_mol(ref_elements, lmax, nmax)

    # training set selection
    nfrac, ntrains, train_configs = get_training_sets(p.trainfilename, o.fracs)

    if len(sys.argv)>1 and sys.argv[1][0].lower()=='b':
        bmatfiles = [f'{p.bmatfilebase}_M{o.M}_trainfrac{frac}.dat' for frac in o.fracs]
        get_b(lmax, nmax, totsize, ref_elements,
              nfrac, ntrains, train_configs,
              p.goodoverfilebase, p.kernelconfbase, bmatfiles)
    else:
        avecfiles = [f'{p.avecfilebase}_M{o.M}_trainfrac{frac}.txt' for frac in o.fracs]
        get_a(lmax, nmax, totsize, ref_elements,
              nfrac, ntrains, train_configs,
              p.baselinedwbase, p.kernelconfbase, avecfiles)


if __name__=='__main__':
    main()
