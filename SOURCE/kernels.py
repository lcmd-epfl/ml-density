#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import *
from power_spectra import read_ps

conf = Config()

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
refsselfilebase = conf.paths['refs_sel_base']
specselfilebase = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
psfilebase      = conf.paths['ps_base']
powerrefbase    = conf.paths['ps_ref_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

#================= SOAP PARAMETERS
zeta = 2.0

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)
(atom_counting, spec_list_per_conf) = get_spec_list_per_conf(species, ndata, natoms, atomic_numbers)

#===================== atomic indices sorted by species
atomicindx = get_atomicindx(ndata, nspecies, natmax, atom_counting, spec_list_per_conf)

#====================================== reference environments
fps_indexes = np.loadtxt(refsselfilebase+str(M)+".txt",int)
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)
llmax = max(lmax.values())

#============================================= PROBLEM DIMENSIONALITY

# load power spectra
power_ref_sparse = {}
power_training = {}
fps_indexes = list(fps_indexes)

for l in range(llmax+1):

    # power spectrum
    (nfeat, power_training[l]) = read_ps(psfilebase+str(l)+".npy", l, ndata, natmax, nspecies, atom_counting, atomicindx)

    if l==0:
        power_ref_sparse[l] = np.zeros((M,nfeat),float)
    else:
        power_ref_sparse[l] = np.zeros((M,2*l+1,nfeat),float)

    ienv = 0
    for iconf in range(ndata):
        for iat in range(natoms[iconf]):
            if ienv in fps_indexes:
                 ind = fps_indexes.index(ienv)
                 power_ref_sparse[l][ind] = power_training[l][iconf,iat]
            ienv += 1
    np.save(powerrefbase+str(l)+"_"+str(M)+".npy", power_ref_sparse[l]);

# compute sparse kernel matrix
for iconf in range(ndata):

    print_progress(iconf, ndata)

    atoms = atomic_numbers[iconf]
    # define sparse indexes
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms[iconf],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in range(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in range(lmax[spe]+1):
            msize = 2*l+1
            for im in range(msize):
                for iat in range(atom_counting[iconf,ispe]):
                    for imm in range(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1
    # compute kernels
    k_NM = np.zeros(kernel_size,float)
    for iref in range(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for iatspe in range(atom_counting[iconf,ispe]):
            iat = atomicindx[iconf,ispe,iatspe]
            ik0 = kernel_sparse_indexes[iref,iatspe,0,0,0]
            for l in range(lmax[spe]+1):
                msize = 2*l+1
                powert = power_training[l][iconf,iat]
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
    np.savetxt(kernelconfbase+str(iconf)+".dat", k_NM,fmt='%.06e')

