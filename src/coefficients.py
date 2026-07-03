#!/usr/bin/env python3

import sys
import numpy as np
import metatensor
from tqdm import tqdm
from libs.config import read_config
from libs.functions import moldata_read, get_test_set, Basis
from libs.tmap import split, tmap2vector, tmap_add


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzfilename)
    averages = metatensor.load(p.spherical_averages)

    basis = Basis(o.basisname, elements=averages.keys.column('center_type'))
    _, test_configs = get_test_set(p.train_test_sets)

    if o.output_coeff_order!='gpr':
        from qstack import compound, reorder
        mols = compound.xyz_to_mol_all(p.xyzfilename, basis=o.basisname, ignore=True)

    for frac in o.fracs:
        print('fraction =', frac)
        predictfile = p.predictions.format(subset='test', train_frac=frac)
        predictions = split(metatensor.load(predictfile))
        for imol, c in zip(tqdm(test_configs), predictions, strict=True):
            tmap_add(c, averages)
            rho = tmap2vector(atomic_numbers[imol], basis, c)
            if o.output_coeff_order!='gpr':
                rho = reorder.reorder_ao(mols[imol], rho, dest=o.output_coeff_order, src='gpr')
            np.savetxt(p.predicted_coeff.format(train_frac=frac, order=o.output_coeff_order, imol=imol), rho)


if __name__=='__main__':
    main()
