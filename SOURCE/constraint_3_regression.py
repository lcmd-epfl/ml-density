#!/usr/bin/env python3

import sys
import numpy as np
from basis import basis_read_full
from config import Config,get_config_path
from functions import moldata_read,get_elements_list,basis_info,averages_read,number_of_electrons_ao,get_baselined_constraints,get_training_sets
import os
import ctypes
import ctypes_def

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    f  = conf.get_option('trainfrac', np.array([1.0]), conf.floats)
    m  = conf.get_option('m'        , 100,             int  )
    r  = conf.get_option('regular'  , 1e-6,            float)
    j  = conf.get_option('jitter'   , 1e-10,           float)
    q  = conf.get_option('charges'  , 0,               int  )
    return [f,m,r,j,q]

[fracs,M,reg,jit,use_charges] = set_variable_values()

kmmbase         = conf.paths['kmm_base']
avecfilebase    = conf.paths['avec_base']
bmatfilebase    = conf.paths['bmat_base']
weightsfilebase = conf.paths['weights_base']
xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
elselfilebase   = conf.paths['spec_sel_base']
chargefilename  = conf.paths['chargesfile']
Kqfilebase      = conf.paths['kernel_charges_base']
avdir           = conf.paths['averages_dir']
trainfilename   = conf.paths['trainingselfile']

kmmfile     = kmmbase+str(M)+".npy"
elselfile   = elselfilebase+str(M)+".txt"
Kqfile      = Kqfilebase+"_M"+str(M)+".dat"

# reference environments
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
print("problem dimensionality =", totsize)

#===============================================================

av_coefs = averages_read(el_dict.values(), avdir)
molcharges = np.loadtxt(chargefilename, dtype=int)
print('charge_file:', chargefilename, 'mode:', use_charges, '\n')

k_MM = np.load(kmmfile)
Kq_full = np.fromfile(Kqfile).reshape(-1,totsize)

regression = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/regression.so")
regression.make_matrix.restype = ctypes.c_int
regression.make_matrix.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_1d_int,
  ctypes_def.array_2d_int,
  ctypes_def.array_3d_double,
  ctypes_def.array_2d_double,
  ctypes.c_double,
  ctypes.c_double,
  ctypes.c_char_p ]

nfrac,ntrains,train_configs_full = get_training_sets(trainfilename, fracs)
constraints_full = get_baselined_constraints(av_coefs, basis, atomic_numbers[train_configs_full], molcharges[train_configs_full], use_charges)

for ntrain,frac in zip(ntrains,fracs):

  print('fraction =', frac)

  avecfile    = avecfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".txt"
  bmatfile    = bmatfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+".dat"
  weightsfile = weightsfilebase+"_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy"

  constraints = constraints_full[:ntrain]
  Kq     = Kq_full[:ntrain]

  Bmat    = np.zeros((totsize,totsize))
  Avec    = np.loadtxt(avecfile)

  ret = regression.make_matrix(
        totsize,
        llmax  ,
        M      ,
        ref_elements.astype(np.uint32),
        alnum.astype(np.uint32),
        annum.astype(np.uint32),
        k_MM, Bmat, reg, jit,
        bmatfile.encode('ascii'))

  x0     = np.linalg.solve(Bmat, Avec)
  B1Kq   = np.linalg.solve(Bmat, Kq.T)
  qKB1Kq = np.einsum('ij,jk->ik', Kq, B1Kq)
  alpha  = np.einsum('ij,j->i', Kq, x0)
  v      = alpha-constraints

  if 1:
    qKB1Kq_reg = qKB1Kq+reg*np.eye(ntrain)
    la = np.linalg.solve(qKB1Kq_reg, v)
    print('cond =', np.linalg.cond(qKB1Kq), np.linalg.cond(qKB1Kq_reg))
  else:
    # why when I use this the constraints are also ok but errors much higher?
    la, residuals, rank, s = np.linalg.lstsq(qKB1Kq, v, rcond=None)
    print('Δrank =', ntrain-rank)

  dx = np.einsum('ij,j->i', B1Kq, la)
  weights = x0 - dx
  np.save(weightsfile, weights)

