#!/usr/bin/env python3

import sys
import numpy as np
from basis import basis_read
from config import Config, get_config_path
from functions import moldata_read, get_elements_list
from libs.kernels_lib import kernel_mm_new


def set_variable_values(conf):
    m   = conf.get_option('m'           ,  100, int  )
    return [m]


def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [M] = set_variable_values(conf)
    xyzfilename     = conf.paths['xyzfile']
    basisfilename   = conf.paths['basisfile']
    kmmbase         = conf.paths['kmm_base']
    powerrefbase    = conf.paths['ps_ref_base']
    refsselfilebase = conf.paths['refs_sel_base']


    _, _, atomic_numbers = moldata_read(xyzfilename)
    elements = get_elements_list(atomic_numbers)

    (el_dict, lmax, nmax) = basis_read(basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)

    ref_indices = np.loadtxt(refsselfilebase+str(M)+".txt", dtype=int)
    ref_elements = np.hstack(atomic_numbers)[ref_indices]
    k_MM = kernel_mm_new(M, lmax, powerrefbase, ref_elements)
    np.save(f'{kmmbase}{M}.npy', k_MM )


if __name__=='__main__':
    main()
