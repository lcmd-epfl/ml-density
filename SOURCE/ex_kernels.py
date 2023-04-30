#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read, get_elements_list, get_atomicindx, print_progress
from libs.kernels_lib import kernel_for_mol


def set_variable_values(conf):
    m   = conf.get_option('m'           ,  100, int  )
    return [m]

def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [M] = set_variable_values(conf)
    xyzfilename   = conf.paths['xyzfile']
    basisfilename = conf.paths['basisfile']
    elselfilebase = conf.paths['spec_sel_base']
    powerrefbase  = conf.paths['ps_ref_base']
    xyzexfilename = conf.paths['ex_xyzfile']
    kernelexbase  = conf.paths['ex_kernel_base']
    powerexbase   = conf.paths['ex_ps_base']


    _, _, atomic_numbers = moldata_read(xyzfilename)
    (nmol_ex, natoms_ex, atomic_numbers_ex) = moldata_read(xyzexfilename)
    natmax_ex = max(natoms_ex)

    # elements array and atomic indices sorted by elements
    elements = get_elements_list(atomic_numbers)
    nel = len(elements)
    (atomicindx_ex, atom_counting_ex, _) = get_atomicindx(elements, atomic_numbers_ex, natmax_ex)

    ref_elements = np.loadtxt(f'{elselfilebase}{M}.txt', dtype=int)

    # elements dictionary, max. angular momenta, number of radial channels
    (el_dict, lmax, _) = basis_read(basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    power_ref = equistore.load(f'{powerrefbase}_{M}.npz');

    for imol in range(nmol_ex):
        print_progress(imol, nmol_ex)
        kernel_for_mol(power_ref, f'{powerexbase}_{imol}.npz', f'{kernelexbase}{imol}.dat',
                       lmax, ref_elements, el_dict,
                       natoms_ex[imol], atom_counting_ex[imol], atomicindx_ex[imol])


if __name__=='__main__':
    main()
