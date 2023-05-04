#!/usr/bin/env python3

import sys
import numpy as np
from config import Config, get_config_path
from basis import basis_read
from functions import moldata_read, get_elements_list, nao_for_mol, get_training_sets
from libs.get_matrices_A import get_a
from libs.get_matrices_B import get_b


def set_variable_values(conf):
    f  = conf.get_option('trainfrac', np.array([1.0]), conf.floats)
    m  = conf.get_option('m'        , 100,             int  )
    return [f, m]


def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [fracs, M] = set_variable_values(conf)
    xyzfilename      = conf.paths['xyzfile']
    basisfilename    = conf.paths['basisfile']
    trainfilename    = conf.paths['trainingselfile']
    refsselfilebase  = conf.paths['refs_sel_base']
    kernelconfbase   = conf.paths['kernel_conf_base']
    baselinedwbase   = conf.paths['baselined_w_base']
    goodoverfilebase = conf.paths['goodover_base']
    avecfilebase     = conf.paths['avec_base']
    bmatfilebase     = conf.paths['bmat_base']

    _, _, atomic_numbers = moldata_read(xyzfilename)
    elements = get_elements_list(atomic_numbers)

    # elements dictionary, max. angular momenta, number of radial channels
    (el_dict, lmax, nmax) = basis_read(basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    # reference environments
    ref_indices = np.loadtxt(f'{refsselfilebase}{M}.txt', dtype=int)
    ref_elements = np.hstack(atomic_numbers)[ref_indices]
    totsize = nao_for_mol(ref_elements, lmax, nmax)

    # training set selection
    fracs.sort()
    nfrac, ntrains, train_configs = get_training_sets(trainfilename, fracs)

    if len(sys.argv)>1 and sys.argv[1][0].lower()=='b':
        bmatfiles= [f'{bmatfilebase}_M{M}_trainfrac{fracs[i]}.dat' for i in range(nfrac)]
        get_b(lmax, nmax, totsize, ref_elements,
              nfrac, ntrains, train_configs,
              goodoverfilebase, kernelconfbase, bmatfiles)
    else:
        avecfiles = [f'{avecfilebase}_M{M}_trainfrac{fracs[i]}.txt' for i in range(nfrac)]
        get_a(lmax, nmax, totsize, ref_elements,
              nfrac, ntrains, train_configs,
              baselinedwbase, kernelconfbase, avecfiles)


if __name__=='__main__':
    main()
