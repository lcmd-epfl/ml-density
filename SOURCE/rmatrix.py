#!/usr/bin/python3

import numpy as np
from basis import basis_read
from config import Config
from functions import *

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
kmmbase         = conf.paths['kmm_base']
powerrefbase    = conf.paths['ps_ref_base']

#================= SOAP PARAMETERS
zeta = 2.0

#==================== species array
(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
species = get_species_list(atomic_numbers)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)
llmax = max(lmax.values())

k_MM = np.zeros((llmax+1,M*(2*llmax+1),M*(2*llmax+1)),float)

for l in range(llmax+1):

    power_ref_sparse = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");

    if l==0:
        for iref1 in range(M):
            for iref2 in range(M):
                k_MM[l,iref1,iref2] = np.dot(power_ref_sparse[iref1],power_ref_sparse[iref2].T)**zeta
    else:
        nfeat = power_ref_sparse.shape[2]
        power_ref_sparse = power_ref_sparse.reshape(M*(2*l+1),nfeat)
        ms = 2*l+1
        for iref1 in range(M):
            for iref2 in range(M):
                k_MM[l, iref1*ms:(iref1+1)*ms, iref2*ms:(iref2+1)*ms] = np.dot(power_ref_sparse[iref1*ms:(iref1+1)*ms], power_ref_sparse[iref2*ms:(iref2+1)*ms].T) *  k_MM[0,iref1,iref2]**((zeta-1.0)/zeta)

np.save(kmmbase+str(M)+".npy", k_MM )

