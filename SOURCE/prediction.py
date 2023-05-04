#!/usr/bin/env python3

import sys
import numpy as np
from config import read_config
from basis import basis_read
from functions import moldata_read, get_elements_list, get_atomicindx, get_test_set, get_training_set
from run_prediction import run_prediction


def main():
    o, p = read_config(sys.argv)
    training = 'training' in sys.argv[1:]

    nmol, natoms, atomic_numbers = moldata_read(p.xyzfilename)
    natmax = max(natoms)
    elements = get_elements_list(atomic_numbers)
    (atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

    for frac in o.fracs:

        if not training:
            ntest, test_configs = get_test_set(p.trainfilename, nmol)
            predictfile = f'{p.predictfilebase}_test_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
        else:
            ntest, test_configs = get_training_set(p.trainfilename, frac)
            predictfile = f'{p.predictfilebase}_training_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'
        weightsfile = f'{p.weightsfilebase}_M{o.M}_trainfrac{frac}_reg{o.reg}_jit{o.jit}.npy'

        natoms_test = natoms[test_configs]
        atomicindx_test = atomicindx[test_configs]
        atom_counting_test = atom_counting[test_configs]

        print(f'Number of testing molecules = {ntest}')

        run_prediction(ntest, natmax, natoms_test,
                       atom_counting_test, atomicindx_test, test_configs, o.M, elements,
                       p.kernelconfbase, p.basisfilename, f'{p.elselfilebase}{o.M}.txt', weightsfile, predictfile)

if __name__=='__main__':
    main()
