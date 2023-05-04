#!/usr/bin/env python3

import sys
import numpy as np
from config import read_config
from basis import basis_read
from functions import moldata_read, get_elements_list
from libs.tmap import merge_ref_ps
import equistore


def main():
    o, p = read_config(sys.argv)

    _, natoms, atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)
    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)

    el_dict, lmax, _ = basis_read(p.basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    ref_mol_at = get_ref_idx(natoms, ref_indices)
    tensor = merge_ref_ps(lmax, elements, atomic_numbers, ref_mol_at, p.splitpsfilebase)
    equistore.save(f'{p.powerrefbase}_{o.M}.npz', tensor)


def get_ref_idx(natoms, refs):
    idx_mol = []
    idx_atm = []
    for imol, nat in enumerate(natoms):
        idx_mol += [imol] * nat
        idx_atm += range(nat)
    ref_imol = np.array(idx_mol)[refs]
    ref_iat  = np.array(idx_atm)[refs]
    return np.vstack((ref_imol, ref_iat)).T


if __name__=='__main__':
    main()
