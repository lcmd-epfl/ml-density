#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, Basis
from libs.predict import run_prediction
from libs.tmap import tmap2vector


def main():
    o, p = read_config(sys.argv)

    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    averages = metatensor.load(p.avfile)
    ref_elements = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)[:,1]
    basis = Basis(o.basisname, elements=set(ref_elements))
    weights = np.load(f'{p.weightsfilebase}_M{o.M}_trainfrac{o.fracs[-1]}_reg{o.reg}_jit{o.jit}.npy')

    predictions = run_prediction(np.arange(len(atomic_numbers_ex)), atomic_numbers_ex,
                                 basis, weights, ref_elements,
                                 p.kernelexbase, averages=averages)

    for imol, (atoms, c) in enumerate(zip(atomic_numbers_ex, predictions, strict=True)):
        np.savetxt(f'{p.outexfilebase}gpr_{imol}.dat', tmap2vector(atoms, basis, c))


if __name__=='__main__':
    main()
