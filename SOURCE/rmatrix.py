#!/usr/bin/python2.7

import numpy as np
import rmatrix
from basis import basis_read
from config import Config
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
kmmbase         = conf.paths['kmm_base']
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

# basis set size
llmax = max(lmax.values())
nnmax = max(nmax.values())
[bsize, almax, anmax] = basis_info(spe_dict, lmax, nmax);

#============================================= PROBLEM DIMENSIONALITY
totsize = sum(bsize[fps_species])
print "problem dimensionality =", totsize

#================================================= TRAINING SETS
k_MM = np.zeros((llmax+1,M*(2*llmax+1),M*(2*llmax+1)),float)

for l in xrange(llmax+1):

    power = np.load(psfilebase+str(l)+".npy")

    # power spectrum
    if l==0:
        nfeat = len(power[0,0])
        power_env = np.zeros((nenv,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,nfeat),float)
    else:
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

    if l==0:
        power_ref_sparse = power_env[fps_indexes]
        for iref1 in xrange(M):
            for iref2 in xrange(M):
                k_MM[l,iref1,iref2] = np.dot(power_ref_sparse[iref1],power_ref_sparse[iref2].T)**zeta
    else:
        power_ref_sparse = power_env[fps_indexes]
        power_ref_sparse = power_ref_sparse.reshape(M*(2*l+1),nfeat)
        ms = 2*l+1
        for iref1 in xrange(M):
            for iref2 in xrange(M):
                k_MM[l, iref1*ms:(iref1+1)*ms, iref2*ms:(iref2+1)*ms] = np.dot(power_ref_sparse[iref1*ms:(iref1+1)*ms], power_ref_sparse[iref2*ms:(iref2+1)*ms].T) *  k_MM[0,iref1,iref2]**((zeta-1.0)/zeta)

Rmat = rmatrix.rmatrix(llmax,nnmax,nspecies,M,totsize,fps_species,almax,anmax,k_MM)

np.save(kmmbase+str(M)+".npy", Rmat)
