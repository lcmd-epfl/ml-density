import numpy as np
import ase.io

def moldata_read(xyzfilename):
  xyzfile = ase.io.read(xyzfilename,":")
  ndata  = len(xyzfile)
  natoms = np.zeros(ndata,int)
  atomic_numbers = []
  for i in range(ndata):
      atomic_numbers.append(xyzfile[i].get_atomic_numbers())
      natoms[i] = len(atomic_numbers[i])
  return (ndata, natoms, atomic_numbers)

def basis_info(spe_dict, lmax, nmax):
    nspecies = len(spe_dict)
    llmax = max(lmax.values())
    bsize = np.zeros(nspecies,int)
    almax = np.zeros(nspecies,int)
    anmax = np.zeros((nspecies,llmax+1),int)
    for ispe in range(nspecies):
        spe = spe_dict[ispe]
        almax[ispe] = lmax[spe]+1
        for l in range(lmax[spe]+1):
            anmax[ispe,l] = nmax[(spe,l)]
            bsize[ispe] += nmax[(spe,l)]*(2*l+1)
    return [bsize, almax, anmax]

def get_kernel_sizes(myrange, fps_species, spe_dict, M, lmax, atom_counting):
    kernel_sizes = np.zeros(len(myrange),int)
    i = 0
    for iconf in myrange:
        for iref in range(M):
            ispe = fps_species[iref]
            spe = spe_dict[ispe]
            temp = 0
            for l in range(lmax[spe]+1):
                msize = 2*l+1
                temp += msize*msize
            kernel_sizes[i] += temp * atom_counting[i,ispe]
        i += 1
    return kernel_sizes

def get_species_list(atomic_numbers):
    return np.sort(list(set(np.array([item for sublist in atomic_numbers for item in sublist]))))

def get_spec_list_per_conf(species, ndata, natoms, atomic_numbers):
    nspecies = len(species)
    spec_list_per_conf = {}
    atom_counting = np.zeros((ndata,nspecies),int)
    for iconf in range(ndata):
        spec_list_per_conf[iconf] = []
        for iat in range(natoms[iconf]):
            for ispe in range(nspecies):
                if atomic_numbers[iconf][iat] == species[ispe]:
                   atom_counting[iconf,ispe] += 1
                   spec_list_per_conf[iconf].append(ispe)
    return (atom_counting, spec_list_per_conf)

def get_atomicindx(ndata,nspecies,natmax,atom_counting,spec_list_per_conf):
    atomicindx = np.zeros((ndata,nspecies,natmax),int)
    for iconf in range(ndata):
        for ispe in range(nspecies):
            indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
            for icount in range(atom_counting[iconf,ispe]):
                atomicindx[iconf,ispe,icount] = indexes[icount]
    return atomicindx

def unravel_weights(M, llmax, nnmax, fps_species, anmax, almax, weights):
    w = np.zeros((M,llmax+1,nnmax,2*llmax+1),float)
    i = 0
    for ienv in range(M):
        ispe = fps_species[ienv]
        al = almax[ispe]
        for l in range(al):
            msize = 2*l+1
            anc = anmax[ispe,l]
            for n in range(anc):
                for im in range(msize):
                    w[ienv,l,n,im] = weights[i]
                    i += 1
    return w

def print_progress(i, n):
    npad = len(str(n))
    strg = "Doing point %*i of %*i (%5.1f %%)"%(npad,i+1,npad,n,100 * float(i+1)/n)
    print(strg, end='\r', flush=True)

def nel_contrib(a):
  # norm = (2.0*a/np.pi)^3/4
  # integral = (pi/a)^3/2
  return pow (2.0*np.pi/a, 0.75)

def number_of_electrons(basis, atoms, c):
  nel = 0.0
  i = 0
  for iat in range(len(atoms)):
    q = atoms[iat]
    for [l,gto] in basis[q]:
      if l==0:
        for p in range(len(gto)):
          a, w = gto[p]
          nel += c[i] * w * nel_contrib(a)
        i+=1
      else:
        i+=2*l+1
  return nel

