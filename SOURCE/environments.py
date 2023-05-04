#!/usr/bin/env python3

import sys
import numpy as np
from config import read_config
from ase.data import chemical_symbols
from functions import moldata_read, get_elements_list, do_fps, get_atomicindx_new, print_progress
from libs.power_spectra_lib import read_ps_1mol_l0


def main():
    o, p = read_config(sys.argv)

    nmol, _, atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)
    element_indices = get_atomicindx_new(elements, atomic_numbers)

    power_env = []
    for imol, atnum in enumerate(atomic_numbers):
        print_progress(imol, nmol)
        power_env.append(read_ps_1mol_l0(f'{p.splitpsfilebase}_{imol}.npz', atnum))
    power_env = np.vstack(power_env)

    ref_indices, distances = do_fps(power_env, o.M)
    ref_elements = np.concatenate(element_indices)[ref_indices]

    np.savetxt(f'{p.refsselfilebase}{o.M}.txt', ref_indices,  fmt='%i')
    np.savetxt(f'{p.elselfilebase}{o.M}.txt',   ref_elements, fmt='%i')

    for i, d in enumerate(distances):
        print(i+1, d)

    for i, q in enumerate(elements):
        n1 = np.count_nonzero(ref_elements==i)
        n2 = np.count_nonzero(np.concatenate(element_indices)==i)
        print(f'# {chemical_symbols[q]}: {n1} / {n2} ({100.0*n1/n2:.1f}%)')

    nuniq = len(np.unique(ref_indices))
    if nuniq != len(ref_indices):
        print(f'warning: I have found only {nuniq} unique environments')


if __name__=='__main__':
    main()
