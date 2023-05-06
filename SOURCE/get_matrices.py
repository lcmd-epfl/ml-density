#!/usr/bin/env python3

import sys
import numpy as np
from config import read_config
from basis import basis_read
from functions import moldata_read, get_elements_list, nao_for_mol, get_training_sets
from libs.get_matrices_A import get_a
from libs.get_matrices_B import get_b


def main():
    o, p = read_config(sys.argv)

    _, _, atomic_numbers = moldata_read(p.xyzfilename)

    # elements dictionary, max. angular momenta, number of radial channels
    elements = get_elements_list(atomic_numbers)
    (el_dict, lmax, nmax) = basis_read(p.basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    # reference environments
    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    ref_elements = np.hstack(atomic_numbers)[ref_indices]
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
