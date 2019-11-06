#!/usr/bin/python

import numpy as np
import ase.io
import argparse
import prediction
from config import Config
from basis import basis_read

conf = Config()

def set_variable_values():
    f   = conf.get_option('trainfrac'   ,  1.0,   float)
    m   = conf.get_option('m'           ,  100,   int  )
    rc  = conf.get_option('cutoffradius',  4.0,   float)
    r   = conf.get_option('regular'     ,  1e-6,  float)
    j   = conf.get_option('jitter'      ,  1e-10, float)
    mol = conf.get_option('molecule'    ,  '',    str  )
    return [f,m,rc,r,j,mol]

[frac,M,rc,reg,jit,mol] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
trainfilename   = conf.paths['trainingselfile']
refsselfilebase = conf.paths['refs_sel_base']
specselfilebase = conf.paths['spec_sel_base']
weightsfilebase = conf.paths['weights_base']
predictfilebase = conf.paths['predict_base']

# system definition
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)

# system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

# atomic species arrays
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)
spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_valence[iconf][iat] == species[ispe]:
               atom_counting[iconf,ispe] += 1
               spec_list_per_conf[iconf].append(ispe)

# atomic indexes sorted by valence
atomicindx = np.zeros((natmax,nspecies,ndata),int)
for iconf in xrange(ndata):
    for ispe in xrange(nspecies):
        indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
        for icount in xrange(atom_counting[iconf,ispe]):
            atomicindx[icount,ispe,iconf] = indexes[icount]


#====================================== reference environments
fps_indexes = np.loadtxt(refsselfilebase+str(M)+".txt",int)
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
llmax = max(lmax.values())
nnmax = max(nmax.values())

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

# dataset partitioning
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
testrange = np.setdiff1d(range(ndata),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]
print "Number of training molecules = ", ntrain
print "Number of testing molecules = ", ntest

# define testing indexes
test_configs = np.array(testrange,int)
atomicindx_test = atomicindx[:,:,testrange]
atom_counting_test = atom_counting[testrange]
test_species = np.zeros((ntest,natmax),int)
for itest in xrange(ntest):
    for iat in xrange(natoms_test[itest]):
        test_species[itest,iat] = spec_list_per_conf[testrange[itest]][iat]

# sparse kernel sizes
kernel_sizes = np.zeros(ntest,int)
itest = 0
for iconf in testrange:
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        temp = 0
        for l in xrange(lmax[spe]+1):
            msize = 2*l+1
            temp += msize*msize
        kernel_sizes[itest] += temp * atom_counting_test[itest,ispe]
    itest += 1

# load regression weights
weights = np.load(weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

# unravel regression weights with explicit indexing
ww = np.zeros((M,llmax+1,nnmax,2*llmax+1),float)
i = 0
for ienv in xrange(M):
    ispe = fps_species[ienv]
    al = almax[ispe]
    for l in xrange(al):
        msize = 2*l+1
        anc = anmax[ispe,l]
        for n in xrange(anc):
            for im in xrange(msize):
                ww[ienv,l,n,im] = weights[i]
                i += 1

# load testing kernels and perform prediction
coeffs = prediction.prediction(mol,kernel_sizes,fps_species,atom_counting_test,atomicindx_test,nspecies,ntest,int(rc),natmax,llmax,nnmax,natoms_test,test_configs,test_species,almax,anmax,M,ww)

np.save(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",coeffs)

