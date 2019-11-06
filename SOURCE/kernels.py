#!/usr/bin/python

import numpy as np
import time
import ase.io
from config import Config
from basis import basis_read
import sys

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


#========================== system definition
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)
#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)
#================= SOAP PARAMETERS
zeta = 2.0
#==================== species array
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)
spec_list = []
spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_valence[iconf][iat] == species[ispe]:
               atom_counting[iconf,ispe] += 1
               spec_list.append(ispe)
               spec_list_per_conf[iconf].append(ispe)
spec_array = np.asarray(spec_list,int)
nenv = len(spec_array)
#===================== atomic indexes sorted by species
atomicindx = np.zeros((ndata,nspecies,natmax),int)
for iconf in xrange(ndata):
    for ispe in xrange(nspecies):
        indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
        for icount in xrange(atom_counting[iconf,ispe]):
            atomicindx[iconf,ispe,icount] = indexes[icount]
#====================================== reference environments
fps_indexes = np.loadtxt(refsselfilebase+str(M)+".txt",int)
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
llmax = max(lmax.values())

#==================================== BASIS SET SIZE ARRAYS
bsize = np.zeros(nspecies,int)
almax = np.zeros(nspecies,int)
anmax = np.zeros((nspecies,llmax+1),int)
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    almax[ispe] = lmax[spe]+1
    for l in xrange(lmax[spe]+1):
        anmax[ispe,l] = nmax[(spe,l)]
        bsize[ispe] += nmax[(spe,l)]*(2*l+1)
#============================================= PROBLEM DIMENSIONALITY
collsize = np.zeros(M,int)
for iref in xrange(1,M):
    collsize[iref] = collsize[iref-1] + bsize[fps_species[iref-1]]
totsize = collsize[-1] + bsize[fps_species[-1]]
print "problem dimensionality =", totsize

# load power spectra
power_ref_sparse = {}
power_training = {}
for l in xrange(llmax+1):

    power = np.load(psfilebase+str(l)+".npy")

    if l==0:

        # power spectrum
        nfeat = len(power[0,0])
        power_env = np.zeros((nenv,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,nfeat),float)
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

    else:

        # power spectrum
        nfeat = len(power[0,0,0])
        power_env = np.zeros((nenv,2*l+1,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,2*l+1,nfeat),float)
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

    start = time.time()
    atoms = atomic_symbols[iconf]
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
#    print time.time()-start, "seconds"

