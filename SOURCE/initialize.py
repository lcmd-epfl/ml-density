#!/usr/bin/python

import numpy as np
import time
import ase
from ase import io
from ase.io import read
from config import Config
from basis import basis_read

conf = Config()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
coefffilebase    = conf.paths['coeff_base']
overfilebase     = conf.paths['over_base']
goodprojfilebase = conf.paths['goodproj_base']
goodoverfilebase = conf.paths['goodover_base']


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

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)

#===================================================== start decomposition
for iconf in xrange(ndata):
    start = time.time()
    print "-------------------------------"
    print "iconf = ", iconf
    coordinates = coords[iconf]
    atoms = atomic_symbols[iconf]
    valences = atomic_valence[iconf]
    #==================================================
    totsize = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            totsize += nmax[(atoms[iat],l)]*(2*l+1)
    #==================================================
    coeffs = np.loadtxt(coefffilebase+str(iconf)+".dat")
    overlap = np.load(overfilebase+str(iconf)+".npy")
    #==================================================
    Coef = np.zeros(totsize,float)
    Over = np.zeros((totsize,totsize),float)
    i1 = 0
    for iat in xrange(natoms[iconf]):
        spe1 = atoms[iat]
        for l1 in xrange(lmax[spe1]+1):
            for n1 in xrange(nmax[(spe1,l1)]):
                for im1 in xrange(2*l1+1):
                    #
                    if l1==1 and im1!=2:
                        Coef[i1] = coeffs[i1+1]
                    elif l1==1 and im1==2:
                        Coef[i1] = coeffs[i1-2]
                    else:
                        Coef[i1] = coeffs[i1]
                    #
                    i2 = 0
                    for jat in xrange(natoms[iconf]):
                        spe2 = atoms[jat]
                        for l2 in xrange(lmax[spe2]+1):
                            for n2 in xrange(nmax[(spe2,l2)]):
                                for im2 in xrange(2*l2+1):
                                    #
                                    if l1==1 and im1!=2 and l2!=1:
                                        Over[i1,i2] = overlap[i1+1,i2]
                                    elif l1==1 and im1==2 and l2!=1:
                                        Over[i1,i2] = overlap[i1-2,i2]
                                    #
                                    elif l2==1 and im2!=2 and l1!=1:
                                        Over[i1,i2] = overlap[i1,i2+1]
                                    elif l2==1 and im2==2 and l1!=1:
                                        Over[i1,i2] = overlap[i1,i2-2]
                                    #
                                    elif l1==1 and im1!=2 and l2==1 and im2!=2:
                                        Over[i1,i2] = overlap[i1+1,i2+1]
                                    elif l1==1 and im1!=2 and l2==1 and im2==2:
                                        Over[i1,i2] = overlap[i1+1,i2-2]
                                    elif l1==1 and im1==2 and l2==1 and im2!=2:
                                        Over[i1,i2] = overlap[i1-2,i2+1]
                                    elif l1==1 and im1==2 and l2==1 and im2==2:
                                        Over[i1,i2] = overlap[i1-2,i2-2]
                                    #
                                    else:
                                        Over[i1,i2] = overlap[i1,i2]
                                    i2 += 1
                    i1 += 1

    Proj = np.dot(Over,Coef)
    ########################################################################################################
    np.save(goodprojfilebase+str(iconf)+".npy",Proj)
    np.save(goodoverfilebase+str(iconf)+".npy",Over)
    print "time =", time.time()-start
