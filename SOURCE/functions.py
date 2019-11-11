import numpy as np
import ase.io

def moldata_read(xyzfilename):
  xyzfile = ase.io.read(xyzfilename,":")
  ndata  = len(xyzfile)
  natoms = np.zeros(ndata,int)
  atomic_numbers = []
  for i in xrange(ndata):
      atomic_numbers.append(xyzfile[i].get_atomic_numbers())
      natoms[i] = len(atomic_numbers[i])
  return (ndata, natoms, atomic_numbers)

def basis_info(spe_dict, lmax, nmax):
    nspecies = len(spe_dict)
    llmax = max(lmax.values())
    bsize = np.zeros(nspecies,int)
    almax = np.zeros(nspecies,int)
    anmax = np.zeros((nspecies,llmax+1),int)
    for ispe in xrange(nspecies):
        spe = spe_dict[ispe]
        almax[ispe] = lmax[spe]+1
        for l in xrange(lmax[spe]+1):
            anmax[ispe,l] = nmax[(spe,l)]
            bsize[ispe] += nmax[(spe,l)]*(2*l+1)
    return [bsize, almax, anmax]

def get_kernel_sizes(myrange, fps_species, spe_dict, M, lmax, atom_counting):
    kernel_sizes = np.zeros(len(myrange),int)
    i = 0
    for iconf in myrange:
        for iref in xrange(M):
            ispe = fps_species[iref]
            spe = spe_dict[ispe]
            temp = 0
            for l in xrange(lmax[spe]+1):
                msize = 2*l+1
                temp += msize*msize
            kernel_sizes[i] += temp * atom_counting[i,ispe]
        i += 1
    return kernel_sizes


def get_spec_list_per_conf(ndata, natoms, atomic_numbers):

    species = np.sort(list(set(np.array([item for sublist in atomic_numbers for item in sublist]))))
    nspecies = len(species)

    spec_list_per_conf = {}
    atom_counting = np.zeros((ndata,nspecies),int)
    for iconf in xrange(ndata):
        spec_list_per_conf[iconf] = []
        for iat in xrange(natoms[iconf]):
            for ispe in xrange(nspecies):
                if atomic_numbers[iconf][iat] == species[ispe]:
                   atom_counting[iconf,ispe] += 1
                   spec_list_per_conf[iconf].append(ispe)
    return (nspecies, atom_counting, spec_list_per_conf)

def get_atomicindx(ndata,nspecies,natmax,atom_counting,spec_list_per_conf):
    atomicindx = np.zeros((ndata,nspecies,natmax),int)
    for iconf in xrange(ndata):
        for ispe in xrange(nspecies):
            indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
            for icount in xrange(atom_counting[iconf,ispe]):
                atomicindx[iconf,ispe,icount] = indexes[icount]
    return atomicindx

