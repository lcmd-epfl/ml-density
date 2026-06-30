#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.data import chemical_symbols
from qstack.mathutils.fps import do_fps
from libs.config import read_config
from libs.functions import moldata_read, get_elements_list
from libs.power_spectra_lib import read_ps_1mol_l0


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)

    power_env, idx_mol, idx_atm = [], [], []
    for imol, atoms in enumerate(tqdm(atomic_numbers)):
        power_env.append(read_ps_1mol_l0(f'{p.splitpsfilebase}_{imol}.mts', atoms))
        idx_mol.append(np.full_like(atoms, imol))
        idx_atm.append(np.arange(len(atoms)))
    power_env = np.vstack(power_env)

    ref_indices, distances = do_fps(power_env, o.M)

    refs = pd.DataFrame({'environment': ref_indices,
                         'q'          : np.hstack(atomic_numbers)[ref_indices],
                         'mol'        : np.hstack(idx_mol)[ref_indices],
                         'atom_in_mol': np.hstack(idx_atm)[ref_indices],
                         'distance'   : np.hstack(([np.nan], distances))})
    refs.to_csv(f'{p.refsselfilebase}{o.M}.csv', index=False)
    print(refs)

    for q in elements:
        n1 = np.count_nonzero(refs['q']==q)
        n2 = np.count_nonzero(np.concatenate(atomic_numbers)==q)
        print(f'# {chemical_symbols[q]}: {n1} / {n2} ({100.0*n1/n2:.1f}%)')

    nuniq = len(np.unique(ref_indices))
    if nuniq != len(ref_indices):
        print(f'Warning: I have found only {nuniq} unique environments')


if __name__=='__main__':
    main()
