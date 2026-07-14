#!/usr/bin/env python3
"""Build reference power spectra from selected environments."""

import numpy as np
import pandas as pd
import metatensor
from libs.config import get_settings
from libs.functions import Basis
from libs.tmap import merge_ref_ps
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()

    refs = pd.read_csv(p.reference_environments)
    ref_mol_at = np.vstack((refs['q'], refs['mol'], refs['atom_in_mol'])).T
    basis = Basis(o.basisname, refs['q'])

    tensor = merge_ref_ps(basis.lmax, ref_mol_at, p.power_spectrum)
    metatensor.save(p.reference_power_spectra, tensor)


if __name__=='__main__':
    main()
