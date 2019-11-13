#!/usr/bin/python

import numpy as np
from config import Config
from functions import *

conf = Config()

def set_variable_values():
    m   = conf.get_option('m'           ,  100, int  )
    return [m]

[M] = set_variable_values()

xyzfilename = conf.paths['xyzfile']
psfilebase  = conf.paths['ps_base']
refsselfilebase = conf.paths['refs_sel_base']
specselfilebase = conf.paths['spec_sel_base']

def do_fps(x, d=0):
    # Code from Giulio Imbalzano
    if d == 0 : d = len(x)
    n = len(x)
    iy = np.zeros(d,int)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum((x*np.conj(x)),axis=1)
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
    for i in xrange(1,d):
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

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
power = np.load(psfilebase+'0.npy')
nfeat = len(power[0,0])
power_env = np.zeros((nenv,nfeat),complex)
ienv = 0
for iconf in xrange(ndata):
    power_per_conf = np.zeros((natoms[iconf],nfeat),complex)
    iat = 0
    for ispe in xrange(nspecies):
        for icount in xrange(atom_counting[iconf,ispe]):
            jat = atomicindx[iconf,ispe,icount]
            power_per_conf[jat,:] = power[iconf,iat,:]
            iat+=1
    for iat in xrange(natoms[iconf]):
        power_env[ienv,:] = power_per_conf[iat,:]
        ienv += 1

fps_indexes = np.array(do_fps(power_env,M),int)
fps_species = spec_array[fps_indexes]
np.savetxt(refsselfilebase+str(M)+".txt",fps_indexes,fmt='%i')
np.savetxt(specselfilebase+str(M)+".txt",fps_species,fmt='%i')

nuniq = len(np.unique(fps_indexes))
if nuniq != len(fps_indexes):
    print 'warning: i have found only ', nuniq, 'unique environments'

