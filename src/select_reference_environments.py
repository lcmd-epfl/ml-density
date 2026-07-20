#!/usr/bin/env python3
"""Select reference environments from the dataset.

Note:
    This selects reference environments from the full dataset
    (both train and test). While our tests showed it is not important,
    in some cases such data leak can lead to wrong results.
"""

import numpy as np
import pandas as pd
from libs.progress import tqdm
from ase.data import chemical_symbols
from qstack.mathutils.fps import do_fps
from libs.config import get_settings
from libs.functions import moldata_read, get_elements
from libs.tmap import read_ps_1mol_l0
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()

    atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements(atomic_numbers)

    power_env, idx_mol, idx_atm = [], [], []
    for imol, atoms in enumerate(tqdm(atomic_numbers)):
        power_env.append(read_ps_1mol_l0(p.power_spectrum.format(imol), atoms))
        idx_mol.append(np.full_like(atoms, imol))
        idx_atm.append(np.arange(len(atoms)))
    power_env = np.vstack(power_env)

    ref_indices, distances = do_fps(power_env, o.M)

    refs = pd.DataFrame({'environment': ref_indices,
                         'q'          : np.hstack(atomic_numbers)[ref_indices],
                         'mol'        : np.hstack(idx_mol)[ref_indices],
                         'atom_in_mol': np.hstack(idx_atm)[ref_indices],
                         'distance'   : np.hstack(([np.nan], distances))})
    refs.to_csv(p.reference_environments, index=False)
    logger.debug(refs)

    for q in elements:
        n1 = np.count_nonzero(refs['q']==q)
        n2 = np.count_nonzero(np.concatenate(atomic_numbers)==q)
        logger.info(f'# {chemical_symbols[q]}: {n1} / {n2} ({100.0*n1/n2:.1f}%)')

    if (nuniq := len(np.unique(ref_indices))) != (nref := len(ref_indices)):
        logger.warning(f'Only {nuniq} / {nref} unique environments found')


if __name__=='__main__':
    main()
