#!/usr/bin/env python3
"""Export predicted electron-density coefficients to output files."""

import numpy as np
import metatensor
from qstack.io.metatensor import split
from qstack import reorder
from tqdm import tqdm
from libs.config import get_settings
from libs.functions import moldata_read, get_test_set, Basis, make_dummy_mol
from libs.tmap import tmap2vector, tmap_add
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()

    atomic_numbers = moldata_read(p.xyzfilename)
    averages = metatensor.load(p.spherical_averages)

    basis = Basis(o.basisname, elements=averages.keys.column('center_type'))
    _, test_configs = get_test_set(p.train_test_sets)

    if o.output_coeff_order!='gpr':
        mols = [make_dummy_mol(atoms, basis=o.basisname, ignore=True) for atoms in atomic_numbers]

    for frac in o.fracs:
        logger.info(f'fraction = {frac}')
        predictfile = p.predictions.format(subset='test', train_frac=frac)
        predictions = split(metatensor.load(predictfile))
        for imol, c in zip(tqdm(test_configs), predictions, strict=True):
            tmap_add(c, averages)
            rho = tmap2vector(atomic_numbers[imol], basis.llist, c)
            if o.output_coeff_order!='gpr':
                rho = reorder.reorder_ao(mols[imol], rho, dest=o.output_coeff_order, src='gpr')
            np.savetxt(p.predicted_coeff.format(train_frac=frac, order=o.output_coeff_order, imol=imol), rho)


if __name__=='__main__':
    main()
