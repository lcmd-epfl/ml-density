#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,get_elements_list,get_el_list_per_conf,get_atomicindx,print_progress
from power_spectra import read_ps_1mol
from kernels_lib import kernel_nm_sparse_indices,kernel_nm

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
power_ref = {}
for l in range(llmax+1):
    power_ref[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");

#------------------------------------------------------------------------

def kernel_for_mol(imol):
    if USEMPI==0:
      print_progress(imol, nmol)
    else:
      print(nproc, ':', imol-start[nproc], flush=1)

    kernel_size, kernel_sparse_indices = kernel_nm_sparse_indices(M, natoms[imol], llmax, lmax, ref_elements, el_dict, atom_counting[imol])

    power_mol = {}
    for l in range(llmax+1):
        (dummy, power_mol[l]) = read_ps_1mol(splitpsfilebase+str(l)+"_"+str(imol)+".npy", nel, atom_counting[imol], atomicindx[imol])

    k_NM = kernel_nm(M, llmax, lmax, nel, el_dict, ref_elements, kernel_size, kernel_sparse_indices, power_mol, power_ref, atom_counting[imol], atomicindx[imol])
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

