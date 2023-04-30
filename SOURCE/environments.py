#!/usr/bin/env python3

import sys
import numpy as np
from config import Config, get_config_path
from ase.data import chemical_symbols
from functions import moldata_read, get_elements_list, do_fps, get_atomicindx_new
from libs.power_spectra_lib import read_ps_1mol_l0


def set_variable_values(conf):
    m   = conf.get_option('m'           ,  100, int  )
    return [m]


def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [M] = set_variable_values(conf)
    xyzfilename     = conf.paths['xyzfile']
    splitpsfilebase = conf.paths['ps_split_base']
    refsselfilebase = conf.paths['refs_sel_base']
    elselfilebase   = conf.paths['spec_sel_base']


    _, _, atomic_numbers = moldata_read(xyzfilename)
    elements = get_elements_list(atomic_numbers)
    element_indices = get_atomicindx_new(elements, atomic_numbers)

    power_env = np.vstack([read_ps_1mol_l0(f'{splitpsfilebase}_{imol}.npz', atnum) for imol, atnum in enumerate(atomic_numbers)])

    ref_indices, distances = do_fps(power_env,M)
    ref_elements = np.concatenate(element_indices)[ref_indices]

    np.savetxt(f'{refsselfilebase}{M}.txt', ref_indices,  fmt='%i')
    np.savetxt(f'{elselfilebase}{M}.txt',   ref_elements, fmt='%i')

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
