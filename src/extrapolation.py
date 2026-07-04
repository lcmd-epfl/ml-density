#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import metatensor
from qstack import reorder
from libs.config import read_config
from libs.functions import moldata_read, Basis, make_dummy_mol
from libs.predict import run_prediction
from libs.tmap import tmap2vector


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzexfilename)
    averages = metatensor.load(p.spherical_averages)
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=ref_elements)
    weights = metatensor.load(p.weights.format(train_frac=o.fracs[-1]))

    predictions = run_prediction(np.arange(len(atomic_numbers)), atomic_numbers,
                                 basis, weights, p.extra_kernel_nm, averages=averages)

    if o.output_coeff_order!='gpr':
        mols = [make_dummy_mol(atoms, basis=o.basisname, ignore=True) for atoms in atomic_numbers]

    for imol, (atoms, c) in enumerate(zip(atomic_numbers, predictions, strict=True)):
        rho = tmap2vector(atoms, basis, c)
        if o.output_coeff_order!='gpr':
            rho = reorder.reorder_ao(mols[imol], rho, dest=o.output_coeff_order, src='gpr')
        np.savetxt(p.extra_predicted_coeff.format(order=o.output_coeff_order, imol=imol), rho)


if __name__=='__main__':
    main()
