#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from libs.config import read_config
from libs.basis import basis_read
from libs.functions import moldata_read, get_elements_list
from libs.tmap import merge_ref_ps


def main():
    o, p = read_config(sys.argv)

    atomic_numbers = moldata_read(p.xyzfilename)
    elements = get_elements_list(atomic_numbers)
    natoms = np.array([len(atoms) for atoms in atomic_numbers])

    ref_indices = np.loadtxt(f'{p.refsselfilebase}{o.M}.txt', dtype=int)
    lmax, _ = basis_read(p.basisfilename)

    ref_mol_at = get_ref_idx(natoms, ref_indices)
    print(ref_mol_at)
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
