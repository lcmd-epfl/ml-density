#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import *
from power_spectra import read_ps_1mol
import time

import os
import sys
import ctypes
import numpy.ctypeslib as npct

conf = Config()

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
specselfilebase = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
powerrefbase    = conf.paths['ps_ref_base']
splitpsfilebase = conf.paths['ps_split_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

#================= SOAP PARAMETERS
zeta = 2.0

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)
(atom_counting, spec_list_per_conf) = get_spec_list_per_conf(species, ndata, natoms, atomic_numbers)

#===================== atomic indices sorted by species
atomicindx = get_atomicindx(ndata, nspecies, natmax, atom_counting, spec_list_per_conf)

#====================================== reference environments
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)
llmax = max(lmax.values())

start = time.time()
# load power spectra
power_ref_sparse = {}
for l in range(llmax+1):
    power_ref_sparse[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");
end = time.time()
print(end - start)

start = time.time()



array_1d_int    = npct.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
array_5d_int    = npct.ndpointer(dtype=np.uint32,  ndim=5, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')
array_3d_double = npct.ndpointer(dtype=np.float64, ndim=3, flags='CONTIGUOUS')
kernels = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/kernels.so")
kernels.kernels.restype = ctypes.c_int




# compute sparse kernel matrix
for iconf in range(1):
#for iconf in range(4):
#for iconf in range(ndata):

    print_progress(iconf, ndata)

    power_training_iconf = {}
    for l in range(llmax+1):
        (dummy, power_training_iconf[l]) = read_ps_1mol(splitpsfilebase+str(l)+"_"+str(iconf)+".npy", nspecies, atom_counting[iconf], atomicindx[iconf])

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
    for iref in range(10):
    #for iref in range(M):
        print(iref)
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for iatspe in range(atom_counting[iconf,ispe]):
            iat = atomicindx[iconf,ispe,iatspe]
            for l in range(lmax[spe]+1):
                msize = 2*l+1
                powert = power_training_iconf[l][iat]
                powerr = power_ref_sparse[l][iref]

                kernels.kernels.argtypes = [
                 ctypes.c_int,
                 ctypes.c_int,
                 ctypes.c_int,
                 ctypes.c_int,
                 ctypes.c_int,
                 ctypes.c_int,
                 ctypes.c_double,
                 array_5d_int,
                 ctypes.c_int,
                 array_1d_double,
                 array_1d_double,
                 array_1d_double,
                 ]

                if l==0:

                  ret = kernels.kernels(
                       M,
                       natoms[iconf],
                       llmax,
                       l,
                       iref,
                       iatspe,
                       zeta,
                       kernel_sparse_indexes.astype(np.uint32),
                       powert.shape[-1],
                       powert.flatten(),
                       powerr.flatten(),
                       k_NM)

                else:
                    ik0 = kernel_sparse_indexes[iref,iatspe,0,0,0]
                    mult = k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in range(msize):
                        for im2 in range(msize):
                            ik = kernel_sparse_indexes[iref,iatspe,l,im2,im1]
                            k_NM[ik] = mult * np.dot(powert[im1],powerr[im2])
    np.savetxt(kernelconfbase+str(iconf)+".dat", k_NM,fmt='%.06e')
end = time.time()
print(end - start)

