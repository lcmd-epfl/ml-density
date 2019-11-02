#!/usr/bin/python

import numpy as np
import ase
from ase import io
from ase.io import read
import argparse
from config import Config
from basis import basis_read

conf = Config()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
coefffilebase    = conf.paths['coeff_base']
overfilebase     = conf.paths['over_base']
goodprojfilebase = conf.paths['goodproj_base']
goodoverfilebase = conf.paths['goodover_base']
avdir            = conf.paths['averages_dir']
baselinedwbase   = conf.paths['baselined_w_base']
overdatbase      = conf.paths['over_dat_base']


bohr2ang = 0.529177249
#========================== system definition
xyzfile = read(xyzfilename,":")
ndata = len(xyzfile)
#======================= system parameters
coords = []
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    coords.append(np.asarray(xyzfile[i].get_positions(),float)/bohr2ang)
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)
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

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)

nenv = {}
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    nenv[spe] = 0
    for iconf in xrange(ndata):
        nenv[spe] += atom_counting[iconf,ispe]
    print spe, nenv[spe]


av_coefs = {}
for spe in spe_dict.values():
    av_coefs[spe] = np.zeros(nmax[(spe,0)],float)

print "computing averages..."
for iconf in xrange(ndata):
    print "iconf = ", iconf
    atoms = atomic_symbols[iconf]
    Proj = np.load(goodprojfilebase+str(iconf)+".npy")
    Over = np.load(goodoverfilebase+str(iconf)+".npy")
    Coef = np.linalg.solve(Over,Proj)
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       av_coefs[spe][n] += Coef[i]
                    i += 1

print "saving averages..."
for spe in spe_dict.values():
    av_coefs[spe] /= nenv[spe]
    np.save(avdir+str(spe)+".npy",av_coefs[spe])

print "computing baselined projections..."
for iconf in xrange(ndata):
    print "iconf = ", iconf
    atoms = atomic_symbols[iconf]
    #==================================================
    totsize = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            totsize += nmax[(atoms[iat],l)]*(2*l+1)
    #==================================================
    Proj = np.load(goodprojfilebase+str(iconf)+".npy")
    Over = np.load(goodoverfilebase+str(iconf)+".npy")
    #==================================================
    Av_coeffs = np.zeros(totsize,float)
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 1
    #==================================================
    Proj -= np.dot(Over,Av_coeffs)
    np.savetxt(baselinedwbase+str(iconf)+".dat",Proj, fmt='%.10e')
    np.savetxt(overdatbase+str(iconf)+".dat", np.concatenate(Over), fmt='%.10e')


