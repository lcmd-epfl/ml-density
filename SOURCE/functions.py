import copy
import numpy as np
import ase.io
from ase.data import chemical_symbols

def moldata_read(xyzfilename):
  xyzfile = ase.io.read(xyzfilename,":")
  nmol  = len(xyzfile)
  natoms = np.zeros(nmol,int)
  atomic_numbers = []
  for i in range(nmol):
      atomic_numbers.append(xyzfile[i].get_atomic_numbers())
      natoms[i] = len(atomic_numbers[i])
  return (nmol, natoms, np.array(atomic_numbers))

def basis_info(el_dict, lmax, nmax):
    nel = len(el_dict)
    llmax = max(lmax.values())
    bsize = np.zeros(nel,int)
    alnum = np.zeros(nel,int)
    annum = np.zeros((nel,llmax+1),int)
    for iel in range(nel):
        q = el_dict[iel]
        alnum[iel] = lmax[q]+1
        for l in range(lmax[q]+1):
            annum[iel,l] = nmax[(q,l)]
            bsize[iel] += nmax[(q,l)]*(2*l+1)
    return [bsize, alnum, annum]

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
    return np.unique(np.concatenate(atomic_numbers))

def get_atomicindx(elements,atomic_numbers,natmax):
    nmol = len(atomic_numbers)
    nel  = len(elements)
    element_indices = []
    atom_counting = np.zeros((nmol,nel),int)
    atomicindx    = np.zeros((nmol,nel,natmax),int)
    for imol,atoms in enumerate(atomic_numbers):
        element_indices.append( np.full(len(atoms),-1,int) )
        for iel,el in enumerate(elements):
            idx = np.nonzero(atomic_numbers[imol]==el)[0]
            count = len(idx)
            atomicindx[imol,iel,0:count] = idx
            atom_counting[imol,iel] = count
            element_indices[imol][idx] = iel
    return (atomicindx, atom_counting, element_indices)

def unravel_weights(M, llmax, nnmax, ref_elements, annum, alnum, weights):
    w = np.zeros((M,llmax+1,nnmax,2*llmax+1),float)
    i = 0
    for ienv in range(M):
        iel = ref_elements[ienv]
        al = alnum[iel]
        for l in range(al):
            msize = 2*l+1
            anc = annum[iel,l]
            for n in range(anc):
                for im in range(msize):
                    w[ienv,l,n,im] = weights[i]
                    i += 1
    return w

def print_progress(i, n):
    npad = len(str(n))
    strg = "Doing point %*i of %*i (%5.1f %%)"%(npad,i+1,npad,n,100 * float(i+1)/n)
    end  = '\r' if i<n-1 else '\n'
    print(strg, end=end, flush=True)

def nel_contrib(a):
  # norm = (2.0*a/np.pi)^3/4
  # integral = (pi/a)^3/2
  return pow (2.0*np.pi/a, 0.75)

def number_of_electrons(basis, atoms, c):
  nel = 0.0
  i = 0
  for q in atoms:
    for [l,gto] in basis[q]:
      if l==0:
        for [a,w] in gto:
          nel += c[i] * w * nel_contrib(a)
        i+=1
      else:
        i+=2*l+1
  return nel

def number_of_electrons_ao(basis, atoms):
  nel = []
  for q in atoms:
    for [l,gto] in basis[q]:
      if l==0:
        t = 0.0
        for [a,w] in gto:
          t += w * nel_contrib(a)
        nel.append(t)
      else:
        nel.extend([0]*(2*l+1))
  return np.array(nel)

def correct_number_of_electrons(c, S, q, N):
  S1q  = np.linalg.solve(S, q)
  return c + S1q * (N - c@q)/(q@S1q)

def averages_read(elements, avdir):
  av_coefs = {}
  for q in elements:
    av_coefs[q] = np.load(avdir+chemical_symbols[q]+".npy")
  return av_coefs

def nao_for_mol(atoms, lmax, nmax):
    nao = 0
    for q in atoms:
      for l in range(lmax[q]+1):
        nao += (2*l+1)*nmax[(q,l)]
    return nao

def prediction2coefficients(atoms, lmax, nmax, coeff, av_coefs):
  size = nao_for_mol(atoms, lmax, nmax)
  rho = np.zeros(size)
  i = 0
  for iat,q in enumerate(atoms):
    for l in range(lmax[q]+1):
      msize = 2*l+1
      for n in range(nmax[(q,l)]):
        if l == 0 :
          rho[i] = coeff[iat,l,n,0] + av_coefs[q][n]
        else:
          rho[i:i+msize] = coeff[iat,l,n,0:msize]
        i+=msize
  return rho

def gpr2pyscf(atoms, lmax, nmax, rho0):
  rho = copy.deepcopy(rho0)
  i = 0
  for iat,q in enumerate(atoms):
    for l in range(lmax[q]+1):
      msize = 2*l+1
      for n in range(nmax[(q,l)]):
        if l == 1:
          rho[i+1] = rho0[i+0]
          rho[i+2] = rho0[i+1]
          rho[i  ] = rho0[i+2]
        i+=msize
  return rho

def get_baselined_constraints(av_coefs, basis, atomic_numbers, molcharges, charges_mode):
  av_charges = np.zeros(max(av_coefs.keys())+1)
  for q in av_coefs.keys():
    ch = number_of_electrons_ao(basis, [q])
    ch = ch[ch!=0.0]
    av_charges[q] = ch @ av_coefs[q]
  constraints = np.zeros(len(atomic_numbers))
  for i,atoms in enumerate(atomic_numbers):
    if charges_mode==1:
      constraints[i] = sum(atoms) - molcharges[i]
    elif charges_mode==2:
      constraints[i] = molcharges[i]
    constraints[i] -= sum(av_charges[atoms])
  return constraints

def get_training_set(filename, fraction=1.0, sort=True):
  train_selection = np.loadtxt(filename, dtype=int, ndmin=1)
  n = int(fraction*len(train_selection))
  train_configs = train_selection[0:n]
  if sort:
    train_configs.sort()
  return n,train_configs

def get_training_sets(filename, fractions):
  train_selection = np.loadtxt(filename, dtype=int, ndmin=1)
  n = (fractions*len(train_selection)).astype(int)
  train_configs = train_selection[0:n[-1]]
  return len(n),n,train_configs

def get_test_set(filename, nmol):
  train_selection = np.loadtxt(filename, dtype=int)
  test_configs = np.setdiff1d(range(nmol),train_selection)
  return len(test_configs),test_configs

