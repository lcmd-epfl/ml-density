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
specselfilebase = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
powerrefbase    = conf.paths['ps_ref_base']
splitpsfilebase = conf.paths['ps_split_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

#================= SOAP PARAMETERS
zeta = 2.0

#==================== species array
species = get_species_list(atomic_numbers)
nspecies = len(species)
(atom_counting, spec_list_per_conf) = get_spec_list_per_conf(species, ndata, natoms, atomic_numbers)

#===================== atomic indices sorted by species
atomicindx = get_atomicindx(ndata, nspecies, natmax, atom_counting, spec_list_per_conf)

#====================================== reference environments
fps_species = np.loadtxt(specselfilebase+str(M)+".txt",int)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)
if list(species) != list(spe_dict.values()):
    print("different elements in the molecules and in the basis:", list(species), "and", list(spe_dict.values()) )
    exit(1)
llmax = max(lmax.values())

# load power spectra
power_ref_sparse = {}
for l in range(llmax+1):
    power_ref_sparse[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");


#------------------------------------------------------------------------

def kernel_for_mol(iconf):
    if USEMPI==0:
      print_progress(iconf, ndata)
    else:
      print(nproc, ':', iconf-start[nproc], flush=1)

    atoms = atomic_numbers[iconf]
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms[iconf],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in range(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in range(lmax[spe]+1):
            msize = 2*l+1
            for im in range(msize):
                for iat in range(atom_counting[iconf,ispe]):
                    for imm in range(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1

    power_training_iconf = {}
    for l in range(llmax+1):
        (dummy, power_training_iconf[l]) = read_ps_1mol(splitpsfilebase+str(l)+"_"+str(iconf)+".npy", nspecies, atom_counting[iconf], atomicindx[iconf])

    # compute kernels
    k_NM = np.zeros(kernel_size,float)
    for iref in range(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for iatspe in range(atom_counting[iconf,ispe]):
            iat = atomicindx[iconf,ispe,iatspe]
            ik0 = kernel_sparse_indexes[iref,iatspe,0,0,0]
            for l in range(lmax[spe]+1):
                msize = 2*l+1
                powert = power_training_iconf[l][iat]
                powerr = power_ref_sparse[l][iref]
                if l==0:
                    ik = kernel_sparse_indexes[iref,iatspe,l,0,0]
                    k_NM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in range(msize):
                        for im2 in range(msize):
                            ik = kernel_sparse_indexes[iref,iatspe,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    np.savetxt(kernelconfbase+str(iconf)+".dat", k_NM,fmt='%.06e')

#------------------------------------------------------------------------
if USEMPI==0:
    for iconf in range(ndata):
        kernel_for_mol(iconf)

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
  div = ndata//Nproc
  rem = ndata%Nproc
  for i in range(0,Nproc):
    start[i+1] = start[i] + ( (div+1) if (i<rem) else div )
  for iconf in range(start[nproc], start[nproc+1]):
      kernel_for_mol(iconf)
  print(nproc, ':', 'finished')

