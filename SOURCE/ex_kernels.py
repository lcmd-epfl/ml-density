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

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
specselfilebase = conf.paths['spec_sel_base']
powerrefbase    = conf.paths['ps_ref_base']
xyzexfilename   = conf.paths['ex_xyzfile']
kernelexbase    = conf.paths['ex_kernel_base']
powerexbase     = conf.paths['ex_ps_base']


(ndata,    natoms,    atomic_numbers   ) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)
(ndata_ex, natoms_ex, atomic_numbers_ex) = moldata_read(xyzexfilename)
natmax_ex = max(natoms_ex)

#================= SOAP PARAMETERS
zeta = 2.0

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)

(atom_counting,    spec_list_per_conf   ) = get_spec_list_per_conf(species, ndata,    natoms,    atomic_numbers   )
(atom_counting_ex, spec_list_per_conf_ex) = get_spec_list_per_conf(species, ndata_ex, natoms_ex, atomic_numbers_ex)

#===================== atomic indices sorted by species
atomicindx    = get_atomicindx(ndata,    nspecies, natmax,    atom_counting,    spec_list_per_conf   )
atomicindx_ex = get_atomicindx(ndata_ex, nspecies, natmax_ex, atom_counting_ex, spec_list_per_conf_ex)

#====================================== reference environments
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)
llmax = max(lmax.values())


## load power spectra
power_ref_sparse = {}
for l in range(llmax+1):
    power_ref_sparse[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");

power_ex = {}
for l in range(llmax+1):
    (nfeat, power_ex[l]) = read_ps(powerexbase+str(l)+".npy", l, ndata_ex, natmax_ex, nspecies, atom_counting_ex, atomicindx_ex)

# compute sparse kernel matrix
for iconf in range(ndata_ex):

    npad = len(str(ndata_ex))
    strg = "Doing point %*i of %*i (%6.2f %%)"%(npad,iconf+1,npad,ndata_ex,100 * float(iconf+1)/ndata_ex)
    sys.stdout.write('%s\r'%strg)
    sys.stdout.flush()

    atoms = atomic_numbers_ex[iconf]
    # define sparse indexes
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms_ex[iconf],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in range(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in range(lmax[spe]+1):
            msize = 2*l+1
            for im in range(msize):
                for iat in range(atom_counting_ex[iconf,ispe]):
                    for imm in range(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1
    # compute kernels
    k_NM = np.zeros(kernel_size,float)
    for iref in range(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for iatspe in range(atom_counting_ex[iconf,ispe]):
            iat = atomicindx_ex[iconf,ispe,iatspe]
            ik0 = kernel_sparse_indexes[iref,iatspe,0,0,0]
            for l in range(lmax[spe]+1):
                msize = 2*l+1
                powert = power_ex[l][iconf,iat]
                powerr = power_ref_sparse[l][iref]
                if l==0:
                    ik = kernel_sparse_indexes[iref,iatspe,l,0,0]
                    k_NM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in range(msize):
                        for im2 in range(msize):
                            ik = kernel_sparse_indexes[iref,iatspe,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    np.savetxt(kernelexbase+str(iconf)+".dat", k_NM,fmt='%.06e')

