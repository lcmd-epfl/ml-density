#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import metatensor
from libs.config import read_config
from libs.functions import moldata_read, get_elements_list, Basis
from libs.tmap import merge_ref_ps


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)

    basis = Basis(o.basisname, elements)

    refs = pd.read_csv(f'{p.refsselfilebase}{o.M}.csv')
    ref_mol_at = np.vstack((refs['q'], refs['mol'], refs['atom_in_mol'])).T

    tensor = merge_ref_ps(basis.lmax, ref_mol_at, p.splitpsfilebase)
    metatensor.save(f'{p.powerrefbase}_{o.M}.mts', tensor)


if __name__=='__main__':
    main()
