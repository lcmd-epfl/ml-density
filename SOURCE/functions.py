import numpy as np
import ase.io

def moldata_read(xyzfilename):
  xyzfile = ase.io.read(xyzfilename,":")
  nmol  = len(xyzfile)
  natoms = np.zeros(nmol,int)
  atomic_numbers = []
  for i in range(nmol):
      atomic_numbers.append(xyzfile[i].get_atomic_numbers())
      natoms[i] = len(atomic_numbers[i])
  return (nmol, natoms, atomic_numbers)

def basis_info(el_dict, lmax, nmax):
    nel = len(el_dict)
    llmax = max(lmax.values())
    bsize = np.zeros(nel,int)
    almax = np.zeros(nel,int)
    anmax = np.zeros((nel,llmax+1),int)
    for iel in range(nel):
        q = el_dict[iel]
        almax[iel] = lmax[q]+1
        for l in range(lmax[q]+1):
            anmax[iel,l] = nmax[(q,l)]
            bsize[iel] += nmax[(q,l)]*(2*l+1)
    return [bsize, almax, anmax]

def get_kernel_sizes(myrange, ref_elements, el_dict, M, lmax, atom_counting):
    kernel_sizes = np.zeros(len(myrange),int)
    i = 0
    for imol in myrange:
        for iref in range(M):
            iel = ref_elements[iref]
            q = el_dict[iel]
            temp = 0
            for l in range(lmax[q]+1):
                msize = 2*l+1
                temp += msize*msize
            kernel_sizes[i] += temp * atom_counting[i,iel]
        i += 1
    return kernel_sizes

def get_elements_list(atomic_numbers):
    return np.sort(list(set(np.array([item for sublist in atomic_numbers for item in sublist]))))

def get_el_list_per_conf(elements, nmol, natoms, atomic_numbers):
    nel = len(elements)
    el_list_per_conf = {}
    atom_counting = np.zeros((nmol,nel),int)
    for imol in range(nmol):
        el_list_per_conf[imol] = []
        for iat in range(natoms[imol]):
            for iel in range(nel):
                if atomic_numbers[imol][iat] == elements[iel]:
                   atom_counting[imol,iel] += 1
                   el_list_per_conf[imol].append(iel)
    return (atom_counting, el_list_per_conf)

def get_atomicindx(nmol,nel,natmax,atom_counting,el_list_per_conf):
    atomicindx = np.zeros((nmol,nel,natmax),int)
    for imol in range(nmol):
        for iel in range(nel):
            indexes = [i for i,x in enumerate(el_list_per_conf[imol]) if x==iel]
            for icount in range(atom_counting[imol,iel]):
                atomicindx[imol,iel,icount] = indexes[icount]
    return atomicindx

def unravel_weights(M, llmax, nnmax, ref_elements, anmax, almax, weights):
    w = np.zeros((M,llmax+1,nnmax,2*llmax+1),float)
    i = 0
    for ienv in range(M):
        iel = ref_elements[ienv]
        al = almax[iel]
        for l in range(al):
            msize = 2*l+1
            anc = anmax[iel,l]
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

