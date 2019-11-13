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
specselfilebase = conf.paths['spec_sel_base']
kmmbase         = conf.paths['kmm_base']
powerrefbase    = conf.paths['ps_ref_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)

#================= SOAP PARAMETERS
zeta = 2.0

#==================== species array
(nspecies, atom_counting, spec_list_per_conf) = get_spec_list_per_conf(ndata, natoms, atomic_numbers)

#====================================== reference environments
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

totsize = sum(bsize[fps_species])
print "problem dimensionality =", totsize

k_MM = np.zeros((llmax+1,M*(2*llmax+1),M*(2*llmax+1)),float)

for l in xrange(llmax+1):

    power_ref_sparse = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");

    if l==0:
        for iref1 in xrange(M):
            for iref2 in xrange(M):
                k_MM[l,iref1,iref2] = np.dot(power_ref_sparse[iref1],power_ref_sparse[iref2].T)**zeta
    else:
        nfeat = power_ref_sparse.shape[2]
        power_ref_sparse = power_ref_sparse.reshape(M*(2*l+1),nfeat)
        ms = 2*l+1
        for iref1 in xrange(M):
            for iref2 in xrange(M):
                k_MM[l, iref1*ms:(iref1+1)*ms, iref2*ms:(iref2+1)*ms] = np.dot(power_ref_sparse[iref1*ms:(iref1+1)*ms], power_ref_sparse[iref2*ms:(iref2+1)*ms].T) *  k_MM[0,iref1,iref2]**((zeta-1.0)/zeta)

Rmat = rmatrix.rmatrix(llmax,nnmax,nspecies,M,totsize,fps_species,almax,anmax,k_MM)
np.save(kmmbase+str(M)+".npy", Rmat)

