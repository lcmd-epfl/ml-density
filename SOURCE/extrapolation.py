#!/usr/bin/env python3

import sys
import numpy as np
from config import read_config
from functions import moldata_read, get_elements_list
from basis import basis_read
from libs.predict import run_prediction
from libs.tmap import tmap2vector
import equistore

def main():
    o, p = read_config(sys.argv)

    _, _, atomic_numbers = moldata_read(p.xyzfilename)
    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    ref_elements= np.hstack(atomic_numbers)[ref_indices]
    elements = set(ref_elements)

    nmol_ex, _, atomic_numbers_ex = moldata_read(p.xyzexfilename)

    elements_ex = get_elements_list(atomic_numbers_ex)
    if not set(elements_ex).issubset(set(elements)):
        print("different elements in the molecule and in the training set:", list(elements_ex), "and", list(elements))
        exit(1)

    _, lmax, nmax = basis_read(p.basisfilename)

    weightsfile = f'{p.weightsfilebase}_M{o.M}_trainfrac{o.fracs[-1]}_reg{o.reg}_jit{o.jit}.npy'
    averages = equistore.load(p.avfile)
    predictions = run_prediction(np.arange(nmol_ex), atomic_numbers_ex, ref_elements,
                                 p.basisfilename, weightsfile, p.kernelexbase, averages=averages)

    for imol, (atoms, c) in enumerate(zip(atomic_numbers_ex, predictions)):
        np.savetxt(f'{p.outexfilebase}gpr_{imol}.dat', tmap2vector(atoms, lmax, nmax, c))


if __name__=='__main__':
    main()
