#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from basis import basis_read_full
from functions import moldata_read,averages_read,print_progress,prediction2coefficients,gpr2pyscf,number_of_electrons_ao,correct_number_of_electrons,get_test_set

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    f   = conf.get_option('trainfrac'   ,  1.0,   float)
    m   = conf.get_option('m'           ,  100,   int  )
    r   = conf.get_option('regular'     ,  1e-6,  float)
    j   = conf.get_option('jitter'      ,  1e-10, float)
    q   = conf.get_option('charges'     ,  0,     int  )  # compatibility
    return [f,m,r,j,q]

[frac,M,reg,jit,use_charges] = set_variable_values()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
trainfilename    = conf.paths['trainingselfile']
predictfilebase  = conf.paths['predict_base']
goodoverfilebase = conf.paths['goodover_base']
avdir            = conf.paths['averages_dir']
outfilebase      = conf.paths['output_base']
if use_charges:
  chargefilename = conf.paths['chargesfile']

# number of molecules, number of atoms in each molecule, atomic numbers
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)

# basis, elements dictionary, max. angular momenta, number of radial channels
(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)

ntest,test_configs = get_test_set(trainfilename, nmol)
coeff = np.load(predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

av_coefs = averages_read(el_dict.values(), avdir)
if use_charges:
  print('charge_file:', chargefilename, 'mode:', use_charges, '\n')
  charges  = np.loadtxt(chargefilename, dtype=int)

for itest,imol in enumerate(test_configs):
  print_progress(itest, ntest)
  atoms = atomic_numbers[imol]

  rho  = prediction2coefficients(atoms, lmax, nmax, coeff[itest], av_coefs)
  rho1 = gpr2pyscf              (atoms, lmax, nmax, rho)
  np.savetxt(outfilebase+'gpr_'  +str(imol)+'.dat', rho)
  np.savetxt(outfilebase+'pyscf_'+str(imol)+'.dat', rho1)

  if use_charges:
    if use_charges == 1:
      N  = sum(atoms) - charges[imol]
    elif use_charges == 2:
      N  = charges[imol]
    S  = np.load(goodoverfilebase+str(imol)+".npy")
    q  = number_of_electrons_ao(basis, atoms)

    rho_n  = correct_number_of_electrons(rho, S, q, N)
    rho_n1 = gpr2pyscf(atoms, lmax, nmax, rho_n)
    np.savetxt(outfilebase+'gpr_'  +str(imol)+'.N.dat', rho_n)
    np.savetxt(outfilebase+'pyscf_'+str(imol)+'.N.dat', rho_n1)

    rho_u = rho + q * (N-q@rho) / (q@q)
    rho_u1 = gpr2pyscf(atoms, lmax, nmax, rho_u)
    np.savetxt(outfilebase+'gpr_'  +str(imol)+'.unit.dat', rho_u)
    np.savetxt(outfilebase+'pyscf_'+str(imol)+'.unit.dat', rho_u1)

    rho_s = rho * N / (q@rho)
    rho_s1 = gpr2pyscf(atoms, lmax, nmax, rho_s)
    np.savetxt(outfilebase+'gpr_'  +str(imol)+'.scaled.dat', rho_s)
    np.savetxt(outfilebase+'pyscf_'+str(imol)+'.scaled.dat', rho_s1)

