#!/usr/bin/env python3

import numpy as np
from basis import basis_read_full
from config import Config
from functions import *
import os
import sys
import ctypes
import numpy.ctypeslib as npct
from kernels_lib import kernel_nm_sparse_indices,kernel_nm

conf = Config()

def set_variable_values():
    f  = conf.get_option('trainfrac'   ,  1.0,   float)
    m  = conf.get_option('m'           ,  100,   int  )
    r  = conf.get_option('regular'     ,  1e-6,  float)
    j  = conf.get_option('jitter'      ,  1e-10, float)
    return [f,m,r,j]

[frac,M,reg,jit] = set_variable_values()

kmmbase         = conf.paths['kmm_base']
avecfilebase    = conf.paths['avec_base']
bmatfilebase    = conf.paths['bmat_base']
weightsfilebase = conf.paths['weights_base']
xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
elselfilebase   = conf.paths['spec_sel_base']
chargefilename   = conf.paths['chargesfile']
Kqfilebase      = 'Kq'
Kqfile          = Kqfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".dat"

kmmfile     = kmmbase+str(M)+".npy"
avecfile    = avecfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".txt"
bmatfile    = bmatfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".dat"
elselfile   = elselfilebase+str(M)+".txt"
weightsfile = weightsfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy"

# print ("  input:")
# print ( xyzfilename )
# print ( basisfilename)
# print ( elselfile )
# print ( kmmfile )
# print ( avecfile )
# print ( bmatfile )
# print ("  output:")
# print (weightsfile)
# print ()

#====================================== reference environments
ref_elements = np.loadtxt(elselfile, int)

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
elements = get_elements_list(atomic_numbers)

# basis, elements dictionary, max. angular momenta, number of radial channels
(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)

# basis set size
llmax = max(lmax.values())
[bsize, alnum, annum] = basis_info(el_dict, lmax, nmax);
totsize = sum(bsize[ref_elements])

#===============================================================

k_MM = np.load(kmmfile)
Avec = np.loadtxt(avecfile)
mat  = np.zeros((totsize,totsize))

array_1d_int    = npct.ndpointer(dtype=np.uint32,  ndim=1, flags='CONTIGUOUS')
array_2d_int    = npct.ndpointer(dtype=np.uint32,  ndim=2, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')
array_3d_double = npct.ndpointer(dtype=np.float64, ndim=3, flags='CONTIGUOUS')
regression = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/regression.so")
regression.make_matrix.restype = ctypes.c_int
regression.make_matrix.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  array_1d_int,
  array_1d_int,
  array_2d_int,
  array_3d_double,
  array_2d_double,
  ctypes.c_double,
  ctypes.c_double,
  ctypes.c_char_p ]

ret = regression.make_matrix(
      totsize,
      llmax  ,
      M      ,
      ref_elements.astype(np.uint32),
      alnum.astype(np.uint32),
      annum.astype(np.uint32),
      k_MM, mat, reg, jit,
      bmatfile.encode('ascii'))

print("problem dimensionality =", totsize)

x0 = np.linalg.solve(mat, Avec)
print( np.sum(np.abs(mat @ x0 - Avec)))



trainfilename    = conf.paths['trainingselfile']
trainrangetot = np.loadtxt(trainfilename,int)
ntrain = int(frac*len(trainrangetot))
train_configs = np.array(sorted(trainrangetot[0:ntrain]))
natmax = max(natoms)
natoms_train = natoms[train_configs]
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

atomicindx_training = atomicindx[train_configs,:,:]
atom_counting_training = atom_counting[train_configs]
atomic_elements = np.zeros((ntrain,natmax),int)
for itrain in range(ntrain):
    for iat in range(natoms_train[itrain]):
        atomic_elements[itrain,iat] = element_indices[train_configs[itrain]][iat]


Kq = np.fromfile(Kqfile)
Kq = Kq.reshape(len(x0),ntrain).T
#Kq = Kq.reshape(ntrain,len(x0))
# idk what's correct

alpha = np.einsum('ij,j->i', Kq, x0)
B1Kq = np.linalg.solve(mat, Kq.T)
qKB1Kq = np.einsum('ij,jk->ik', Kq, B1Kq)

charges = np.loadtxt(chargefilename, dtype=int)
charges = charges[train_configs]

la = np.linalg.solve(qKB1Kq+reg*np.eye(ntrain), alpha-charges)

dx = np.einsum('ij,j->i', B1Kq, la)
weights = x0 - dx

np.save(weightsfile, weights)
