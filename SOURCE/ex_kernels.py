#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,get_elements_list,get_el_list_per_conf,get_atomicindx,print_progress
from power_spectra import read_ps
from kernels_lib import kernel_nm_sparse_indices,kernel_nm

conf = Config()

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename   = conf.paths['xyzfile']
basisfilename = conf.paths['basisfile']
elselfilebase = conf.paths['spec_sel_base']
powerrefbase  = conf.paths['ps_ref_base']
xyzexfilename = conf.paths['ex_xyzfile']
kernelexbase  = conf.paths['ex_kernel_base']
powerexbase   = conf.paths['ex_ps_base']


(nmol,    natoms,    atomic_numbers   ) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)
(nmol_ex, natoms_ex, atomic_numbers_ex) = moldata_read(xyzexfilename)
natmax_ex = max(natoms_ex)

#==================== elements array
elements = get_elements_list(atomic_numbers)
nel = len(elements)

(atom_counting,    el_list_per_conf   ) = get_el_list_per_conf(elements, nmol,    natoms,    atomic_numbers   )
(atom_counting_ex, el_list_per_conf_ex) = get_el_list_per_conf(elements, nmol_ex, natoms_ex, atomic_numbers_ex)

#===================== atomic indices sorted by elements
atomicindx    = get_atomicindx(nmol,    nel, natmax,    atom_counting,    el_list_per_conf   )
atomicindx_ex = get_atomicindx(nmol_ex, nel, natmax_ex, atom_counting_ex, el_list_per_conf_ex)

#====================================== reference environments
ref_elements = np.loadtxt(elselfilebase+str(M)+".txt",int)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)
llmax = max(lmax.values())

#=============================================================

power_ref = {}
for l in range(llmax+1):
    power_ref[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");

power_ex = {}
for l in range(llmax+1):
    (nfeat, power_ex[l]) = read_ps(powerexbase+str(l)+".npy", nmol_ex, nel, atom_counting_ex, atomicindx_ex)

for imol in range(nmol_ex):
    print_progress(imol, nmol_ex)
    kernel_size, kernel_sparse_indices = kernel_nm_sparse_indices(M, natoms_ex[imol], llmax, lmax, ref_elements, el_dict, atom_counting_ex[imol])
    k_NM = kernel_nm(M, llmax, lmax, nel, el_dict, ref_elements, kernel_size, kernel_sparse_indices, power_ex, power_ref, atom_counting_ex[imol], atomicindx_ex[imol], imol)
    np.savetxt(kernelexbase+str(imol)+".dat", k_NM,fmt='%.06e')

