#!/usr/bin/python

import numpy as np
from config import Config
from basis import basis_read
import sys
from functions import *

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


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

#================= SOAP PARAMETERS
zeta = 2.0

#==================== species array
(nspecies, atom_counting, spec_list_per_conf) = get_spec_list_per_conf(ndata, natoms, atomic_numbers)

#===================== atomic indices sorted by species
atomicindx = get_atomicindx(ndata,nspecies,natmax,atom_counting,spec_list_per_conf)

#====================================== reference environments
fps_indexes = np.loadtxt(refsselfilebase+str(M)+".txt",int)
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if len(spe_dict) != nspecies:
    print "different number of elements in the molecules and in the basis"
    exit(1)
llmax = max(lmax.values())

#============================================= PROBLEM DIMENSIONALITY

# load power spectra
power_ref_sparse = {}
power_training = {}
for l in xrange(llmax+1):

    power = np.load(psfilebase+str(l)+".npy")

    if l==0:
        nfeat = len(power[0,0])
        power_env = np.zeros((nenv,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,nfeat),float)
    else:
        nfeat = len(power[0,0,0])
        power_env = np.zeros((nenv,2*l+1,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,2*l+1,nfeat),float)

    # power spectrum
    ienv = 0
    for iconf in xrange(ndata):
        iat = 0
        for ispe in xrange(nspecies):
            for icount in xrange(atom_counting[iconf,ispe]):
                jat = atomicindx[iconf,ispe,icount]
                power_per_conf[iconf,jat] = power[iconf,iat]
                iat+=1
        for iat in xrange(natoms[iconf]):
            power_env[ienv] = power_per_conf[iconf,iat]
            ienv += 1
    power_ref_sparse[l] = power_env[fps_indexes]
    power_training[l] = power_per_conf

# compute sparse kernel matrix
for iconf in xrange(ndata):

    npad = len(str(ndata))
    strg = "Doing point %*i of %*i (%6.2f %%)"%(npad,iconf+1,npad,ndata,100 * float(iconf+1)/ndata)
    sys.stdout.write('%s\r'%strg)
    sys.stdout.flush()

    atoms = atomic_numbers[iconf]
    # define sparse indexes
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms[iconf],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in xrange(lmax[spe]+1):
            msize = 2*l+1
            for im in xrange(msize):
                for iat in xrange(atom_counting[iconf,ispe]):
                    for imm in xrange(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1
    # compute kernels
    k_NM = np.zeros(kernel_size,float)
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for iatspe in xrange(atom_counting[iconf,ispe]):
            iat = atomicindx[iconf,ispe,iatspe]
            ik0 = kernel_sparse_indexes[iref,iatspe,0,0,0]
            for l in xrange(lmax[spe]+1):
                msize = 2*l+1
                powert = power_training[l][iconf,iat]
                powerr = power_ref_sparse[l][iref]
                if l==0:
                    ik = kernel_sparse_indexes[iref,iatspe,l,0,0]
                    k_NM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in xrange(msize):
                        for im2 in xrange(msize):
                            ik = kernel_sparse_indexes[iref,iatspe,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    np.savetxt(kernelconfbase+str(iconf)+".dat", k_NM,fmt='%.06e')

