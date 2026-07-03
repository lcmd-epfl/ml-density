#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import metatensor
from libs.config import read_config
from libs.functions import Basis
from libs.tmap import merge_ref_ps


def main():
    o, p = read_config(sys.argv)

    refs = pd.read_csv(p.reference_environments)
    ref_mol_at = np.vstack((refs['q'], refs['mol'], refs['atom_in_mol'])).T
    basis = Basis(o.basisname, refs['q'])

    tensor = merge_ref_ps(basis.lmax, ref_mol_at, p.power_spectrum)
    metatensor.save(p.reference_power_spectra, tensor)


if __name__=='__main__':
    main()
