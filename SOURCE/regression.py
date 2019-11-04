#!/usr/bin/python

import numpy as np
import ase.io
from config import Config
from basis import basis_read

conf = Config()

def set_variable_values():
    s  = conf.get_option('testset'     ,  1,     int  )
    f  = conf.get_option('trainfrac'   ,  1.0,   float)
    m  = conf.get_option('m'           ,  100,   int  )
    rc = conf.get_option('cutoffradius',  4.0,   float)
    sg = conf.get_option('sigmasoap'   ,  0.3,   float)
    r  = conf.get_option('regular'     ,  1e-6,  float)
    j  = conf.get_option('jitter'      ,  1e-10, float)
    return [s,f,m,rc,sg,r,j]

[nset,frac,M,rc,sigma_soap,reg,jit] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
refsselfilebase = conf.paths['refs_sel_base']
specselfilebase = conf.paths['spec_sel_base']
kmmbase         = conf.paths['kmm_base']
avecfilebase    = conf.paths['avec_base']
bmatfilebase    = conf.paths['bmat_base']
weightsfilebase = conf.paths['weights_base']

# coversion factors
bohr2ang = 0.529177249

# system definition
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)

# system parameters
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
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)

#====================================== reference environments
fps_indexes = np.loadtxt(refsselfilebase+str(M)+".txt",int)
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
llmax = max(lmax.values())

# basis set arrays
bsize = np.zeros(nspecies,int)
almax = np.zeros(nspecies,int)
anmax = np.zeros((nspecies,llmax+1),int)
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    almax[ispe] = lmax[spe]+1
    for l in xrange(lmax[spe]+1):
        anmax[ispe,l] = nmax[(spe,l)]
        bsize[ispe] += nmax[(spe,l)]*(2*l+1)

# problem dimensionality
collsize = np.zeros(M,int)
for iref in xrange(1,M):
    collsize[iref] = collsize[iref-1] + bsize[fps_species[iref-1]]
totsize = collsize[-1] + bsize[fps_species[-1]]
print "problem dimensionality =", totsize

Avec = np.load(avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".npy")
Bmat = np.load(bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".npy")
Rmat = np.load(kmmbase+str(M)+".npy")

# solve the regularized sparse regression problem
weights = np.linalg.solve(Bmat + reg*Rmat + jit*np.eye(totsize),Avec)

np.save(weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",weights)

