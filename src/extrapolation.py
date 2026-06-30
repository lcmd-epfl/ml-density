#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, Basis
from libs.predict import run_prediction
from libs.tmap import tmap2vector


def main():
    o, p = read_config(sys.argv)

    atomic_numbers_ex = moldata_read(p.xyzexfilename)
    averages = metatensor.load(p.avfile)
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=set(ref_elements))
    weights = np.load(p.weights.format(train_frac=o.fracs[-1]))

    predictions = run_prediction(np.arange(len(atomic_numbers_ex)), atomic_numbers_ex,
                                 basis, weights, ref_elements,
                                 p.kernelexbase, averages=averages)

    for imol, (atoms, c) in enumerate(zip(atomic_numbers_ex, predictions, strict=True)):
        np.savetxt(f'{p.outexfilebase}gpr_{imol}.dat', tmap2vector(atoms, basis, c))


if __name__=='__main__':
    main()
