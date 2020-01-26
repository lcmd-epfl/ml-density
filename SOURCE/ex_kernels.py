#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
import sys
from functions import *
from power_spectra import read_ps

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

#================= SOAP PARAMETERS
zeta = 2.0

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


## load power spectra
power_ref_sparse = {}
for l in range(llmax+1):
    power_ref_sparse[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");

power_ex = {}
for l in range(llmax+1):
    (nfeat, power_ex[l]) = read_ps(powerexbase+str(l)+".npy", nmol_ex, nel, atom_counting_ex, atomicindx_ex)

# compute sparse kernel matrix
for imol in range(nmol_ex):

    npad = len(str(nmol_ex))
    strg = "Doing point %*i of %*i (%6.2f %%)"%(npad,imol+1,npad,nmol_ex,100 * float(imol+1)/nmol_ex)
    sys.stdout.write('%s\r'%strg)
    sys.stdout.flush()

    atoms = atomic_numbers_ex[imol]
    # define sparse indexes
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms_ex[imol],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in range(M):
        iel = ref_elements[iref]
        q = el_dict[iel]
        for l in range(lmax[q]+1):
            msize = 2*l+1
            for im in range(msize):
                for iat in range(atom_counting_ex[imol,iel]):
                    for imm in range(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1
    # compute kernels
    k_NM = np.zeros(kernel_size,float)
    for iref in range(M):
        iel = ref_elements[iref]
        q = el_dict[iel]
        for iatq in range(atom_counting_ex[imol,iel]):
            iat = atomicindx_ex[imol,iel,iatq]
            ik0 = kernel_sparse_indexes[iref,iatq,0,0,0]
            for l in range(lmax[q]+1):
                msize = 2*l+1
                powert = power_ex[l][imol,iat]
                powerr = power_ref_sparse[l][iref]
                if l==0:
                    ik = kernel_sparse_indexes[iref,iatq,l,0,0]
                    k_NM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in range(msize):
                        for im2 in range(msize):
                            ik = kernel_sparse_indexes[iref,iatq,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    np.savetxt(kernelexbase+str(imol)+".dat", k_NM,fmt='%.06e')

