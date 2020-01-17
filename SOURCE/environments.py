#!/usr/bin/python3

import numpy as np
from config import Config
from ase.data import chemical_symbols
from functions import *
from power_spectra import reorder_ps

conf = Config()

def set_variable_values():
    m   = conf.get_option('m'           ,  100, int  )
    return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
ps0file         = conf.paths['ps0file']
refsselfilebase = conf.paths['refs_sel_base']
specselfilebase = conf.paths['spec_sel_base']

def do_fps(x, d=0):
    # Code from Giulio Imbalzano
    n = len(x)
    if d==0:
        d = n
    iy = np.zeros(d,int)
    measure = np.zeros(d-1,float)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum(x*x, axis=1)
    dl = n2 + n2[iy[0]] - 2.0*np.dot(x,x[iy[0]])
    for i in range(1,d):
        iy[i], measure[i-1] = np.argmax(dl), np.amax(dl)
        nd = n2 + n2[iy[i]] - 2.0*np.dot(x,x[iy[i]])
        dl = np.minimum(dl,nd)
    return iy, measure

# number of molecules, number of atoms in each molecule, atomic numbers
(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)
(atom_counting, spec_list_per_conf) = get_spec_list_per_conf(species, ndata, natoms, atomic_numbers)

spec_list = []
for i in spec_list_per_conf.values():
  spec_list += i
spec_array = np.asarray(spec_list,int)

#===================== atomic indexes sorted by species
atomicindx = get_atomicindx(ndata,nspecies,natmax,atom_counting,spec_list_per_conf)

#====================== environmental power spectrum
power = np.load(ps0file)
nfeat = power.shape[-1]
power_env = np.zeros((nenv,nfeat),float)
ienv = 0
for iconf in range(ndata):
    reorder_ps(power_env[ienv:ienv+natoms[iconf]], power[iconf], nspecies, atom_counting[iconf], atomicindx[iconf])
    ienv += natoms[iconf]

fps_indexes, measure = do_fps(power_env,M)
fps_species = spec_array[fps_indexes]
np.savetxt(refsselfilebase+str(M)+".txt",fps_indexes,fmt='%i')
np.savetxt(specselfilebase+str(M)+".txt",fps_species,fmt='%i')

for i in range(1,M):
  print(i, measure[i-1])

fps_species_list = fps_species.tolist()
for i in range(nspecies):
  n1 = fps_species_list.count(i)
  n2 = spec_list.count(i)
  print('#', chemical_symbols[species[i]]+':', n1, '/', n2, "(%.1f%%)"%(100.0*n1/n2) )

nuniq = len(np.unique(fps_indexes))
if nuniq != len(fps_indexes):
    print('warning: i have found only', nuniq, 'unique environments')

