#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read, get_elements_list, get_atomicindx, print_progress
from libs.kernels_lib import kernel_for_mol
from libs.multi import multi_process

USEMPI = 1


def set_variable_values(conf):
    m   = conf.get_option('m'           ,  100, int  )
    return [m]

def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [M] = set_variable_values(conf)
    xyzfilename     = conf.paths['xyzfile']
    basisfilename   = conf.paths['basisfile']
    elselfilebase   = conf.paths['spec_sel_base']
    kernelconfbase  = conf.paths['kernel_conf_base']
    powerrefbase    = conf.paths['ps_ref_base']
    splitpsfilebase = conf.paths['ps_split_base']


    def do_mol(imol):
        kernel_for_mol(power_ref, f'{splitpsfilebase}_{imol}.npz', f'{kernelconfbase}{imol}.dat',
                       lmax, ref_elements, el_dict,
                       natoms[imol], atom_counting[imol], atomicindx[imol])

    (nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
    natmax = max(natoms)

    # elements array and atomic indices sorted by elements
    elements = get_elements_list(atomic_numbers)
    nel = len(elements)
    atomicindx, atom_counting, _ = get_atomicindx(elements, atomic_numbers, natmax)

    ref_elements = np.loadtxt(f'{elselfilebase}{M}.txt', dtype=int)
    power_ref = equistore.load(f'{powerrefbase}_{M}.npz');

    (el_dict, lmax, nmax) = basis_read(basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)
    llmax = max(lmax.values())


    if USEMPI==0:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        multi_process(nmol, do_mol)


if __name__=='__main__':
    main()
