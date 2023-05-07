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

    ref_elements = np.loadtxt(f'{p.qrefsselfilebase}{o.M}.txt', dtype=int)
    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    lmax, nmax = basis_read(p.basisfilename)
    averages = equistore.load(p.avfile)

    weightsfile = f'{p.weightsfilebase}_M{o.M}_trainfrac{o.fracs[-1]}_reg{o.reg}_jit{o.jit}.npy'
    predictions = run_prediction(np.arange(len(atomic_numbers_ex)), atomic_numbers_ex, ref_elements,
                                 p.basisfilename, weightsfile, p.kernelexbase, averages=averages)

    for imol, (atoms, c) in enumerate(zip(atomic_numbers_ex, predictions)):
        np.savetxt(f'{p.outexfilebase}gpr_{imol}.dat', tmap2vector(atoms, lmax, nmax, c))


if __name__=='__main__':
    main()
