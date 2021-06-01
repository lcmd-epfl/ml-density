#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from functions import moldata_read,get_elements_list,get_atomicindx,prediction2coefficients,gpr2pyscf,number_of_electrons,averages_read,number_of_electrons_ao
from basis import basis_read_full
from run_prediction import run_prediction

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    f   = conf.get_option('trainfrac'   ,  1.0,   float)
    m   = conf.get_option('m'           ,  100,   int  )
    r   = conf.get_option('regular'     ,  1e-6,  float)
    j   = conf.get_option('jitter'      ,  1e-10, float)
    return [f,m,r,j]

[frac,M,reg,jit] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
elselfilebase   = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
weightsfilebase = conf.paths['weights_base']
avdir           = conf.paths['averages_dir']
xyzexfilename   = conf.paths['ex_xyzfile']
kernelexbase    = conf.paths['ex_kernel_base']
predictfilebase = conf.paths['ex_predict_base']
outfilebase     = conf.paths['ex_output_base']

if 'ex_charges' in conf.paths:
  correct_n      = True
  exchargesfiles = conf.paths['ex_charges']
  charges        = np.loadtxt(exchargesfiles, dtype=int)
  if charges.shape==():
    charges = [int(charges)]
else:
  correct_n      = False

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
(nmol_ex, natoms_ex, atomic_numbers_ex) = moldata_read(xyzexfilename)
natmax_ex = max(natoms_ex)

# elements array
elements = get_elements_list(atomic_numbers)
elements_ex = get_elements_list(atomic_numbers_ex)
if not set(elements_ex).issubset(set(elements)):
    print("different elements in the molecule and in the training set:", list(elements_ex), "and", list(elements))
    exit(1)

(atomicindx_ex, atom_counting_ex, element_indices_ex) = get_atomicindx(elements, atomic_numbers_ex, natmax_ex)
test_configs = np.arange(nmol_ex)

pred = run_prediction(nmol_ex, natmax_ex, natoms_ex,
    atom_counting_ex, atomicindx_ex, test_configs,
    M, elements,
    kernelexbase,
    basisfilename,
    elselfilebase+str(M)+".txt",
    weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",
    predictfilebase + "_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

(basis, el_dict, lmax, nmax) = basis_read_full(basisfilename)
av_coefs = averages_read(elements_ex, avdir)
for imol in range(nmol_ex):
    atoms = atomic_numbers_ex[imol]
    nel_exact = sum(atoms)
    if correct_n!=None:
      nel_exact -= charges[imol]

    rho  = prediction2coefficients(atoms, lmax, nmax, pred[imol], av_coefs)
    rho1 = gpr2pyscf              (atoms, lmax, nmax, rho)
    nel = number_of_electrons(basis, atoms, rho1)
    strg = "mol # %*i :  %8.4f / %3d (%.1e)"%(
        len(str(nmol_ex)), imol, nel, nel_exact, nel-nel_exact )
    print(strg)
    np.savetxt(outfilebase+'gpr_'  +str(imol)+'.dat', rho)
    np.savetxt(outfilebase+'pyscf_'+str(imol)+'.dat', rho1)

    if correct_n!=None:
      q  = number_of_electrons_ao(basis, atoms)
      rho_u = rho + q * (nel_exact-q@rho) / (q@q)
      rho_u1 = gpr2pyscf(atoms, lmax, nmax, rho_u)
      np.savetxt(outfilebase+'gpr_'  +str(imol)+'.unit.dat', rho_u)
      np.savetxt(outfilebase+'pyscf_'+str(imol)+'.unit.dat', rho_u1)

