#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import *
from power_spectra import read_ps_1mol

USEMPI = 1

conf = Config()

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
elselfilebase   = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
powerrefbase    = conf.paths['ps_ref_base']
splitpsfilebase = conf.paths['ps_split_base']


(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

#================= SOAP PARAMETERS
zeta = 2.0

elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atom_counting, el_list_per_conf) = get_el_list_per_conf(elements, nmol, natoms, atomic_numbers)

#===================== atomic indices sorted by elements
atomicindx = get_atomicindx(nmol, nel, natmax, atom_counting, el_list_per_conf)

#====================================== reference environments
ref_elements = np.loadtxt(elselfilebase+str(M)+".txt",int)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)
llmax = max(lmax.values())

# load power spectra
power_ref_sparse = {}
for l in range(llmax+1):
    power_ref_sparse[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");


#------------------------------------------------------------------------

def kernel_for_mol(imol):
    if USEMPI==0:
      print_progress(imol, nmol)
    else:
      print(nproc, ':', imol-start[nproc], flush=1)

    atoms = atomic_numbers[imol]
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms[imol],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in range(M):
        iel = ref_elements[iref]
        q   = el_dict[iel]
        for l in range(lmax[q]+1):
            msize = 2*l+1
            for im in range(msize):
                for iat in range(atom_counting[imol,iel]):
                    for imm in range(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1

    power_training_imol = {}
    for l in range(llmax+1):
        (dummy, power_training_imol[l]) = read_ps_1mol(splitpsfilebase+str(l)+"_"+str(imol)+".npy", nel, atom_counting[imol], atomicindx[imol])

    # compute kernels
    k_NM = np.zeros(kernel_size,float)
    for iref in range(M):
        iel = ref_elements[iref]
        q   = el_dict[iel]
        for iatq in range(atom_counting[imol,iel]):
            iat = atomicindx[imol,iel,iatq]
            ik0 = kernel_sparse_indexes[iref,iatq,0,0,0]
            for l in range(lmax[q]+1):
                msize = 2*l+1
                powert = power_training_imol[l][iat]
                powerr = power_ref_sparse[l][iref]
                if l==0:
                    ik = kernel_sparse_indexes[iref,iatq,l,0,0]
                    k_NM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in range(msize):
                        for im2 in range(msize):
                            ik = kernel_sparse_indexes[iref,iatq,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    np.savetxt(kernelconfbase+str(imol)+".dat", k_NM,fmt='%.06e')

#------------------------------------------------------------------------
if USEMPI==0:
    for imol in range(nmol):
        kernel_for_mol(imol)

else:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  Nproc = comm.Get_size()
  nproc = comm.Get_rank()

  msg = "proc " + "%3d"%(nproc) + ' : ' + MPI.Get_processor_name()
  if nproc == 0:
    print(msg)
    for i in range(1, Nproc):
      msg = comm.recv(source=i)
      print (msg)
    print(flush=1)
  else:
    comm.send(msg, dest=0)
  comm.barrier()

  start = [0]*(Nproc+1)
  div = nmol//Nproc
  rem = nmol%Nproc
  for i in range(0,Nproc):
    start[i+1] = start[i] + ( (div+1) if (i<rem) else div )
  for imol in range(start[nproc], start[nproc+1]):
      kernel_for_mol(imol)
  print(nproc, ':', 'finished')

