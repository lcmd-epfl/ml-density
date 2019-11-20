#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import *
from power_spectra import read_ps_1mol

conf = Config()

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
refsselfilebase = conf.paths['refs_sel_base']
powerrefbase    = conf.paths['ps_ref_base']
splitpsfilebase = conf.paths['ps_split_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)
(atom_counting, spec_list_per_conf) = get_spec_list_per_conf(species, ndata, natoms, atomic_numbers)

#===================== atomic indices sorted by species
atomicindx = get_atomicindx(ndata, nspecies, natmax, atom_counting, spec_list_per_conf)

#====================================== reference environments
fps_indexes = np.loadtxt(refsselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)
llmax = max(lmax.values())


fps_indexes = list(fps_indexes)
ref_iconf = [None]*M
ref_iat   = [None]*M
ienv = 0
for iconf in range(ndata):
    for iat in range(natoms[iconf]):
        if ienv in fps_indexes:
             ind = fps_indexes.index(ienv)
             ref_iconf[ind] = iconf
             ref_iat  [ind] = iat
        ienv += 1

for l in range(llmax+1):

    print(" l =", l)
    power_training = {}
    for iconf in set(ref_iconf):
        ( nfeat, power_training[iconf] ) = read_ps_1mol(splitpsfilebase+str(l)+"_"+str(iconf)+".npy", nspecies, atom_counting[iconf], atomicindx[iconf])

    if l==0:
        power_ref_sparse = np.zeros((M,nfeat),float)
    else:
        power_ref_sparse = np.zeros((M,2*l+1,nfeat),float)

    for ind in range(M):
        iconf = ref_iconf[ind]
        iat   = ref_iat  [ind]
        power_ref_sparse[ind] = power_training[iconf][iat]

    np.save(powerrefbase+str(l)+"_"+str(M)+".npy", power_ref_sparse);

