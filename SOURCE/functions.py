import copy
import numpy as np
import ase.io
from ase.data import chemical_symbols


def moldata_read(xyzfilename):
    mols = ase.io.read(xyzfilename, ":")
    nmol = len(mols)
    natoms = np.zeros(nmol, int)
    atomic_numbers = []
    for i, mol in enumerate(mols):
        atomic_numbers.append(mol.get_atomic_numbers())
        natoms[i] = mol.get_global_number_of_atoms()
    return (nmol, natoms, np.array(atomic_numbers, dtype=object))


def get_elements_list(atomic_numbers, return_counts=False):
    return np.unique(np.concatenate(atomic_numbers), return_counts=return_counts)


def print_progress(i, n):
    npad = len(str(n))
    strg = "Doing point %*i of %*i (%5.1f %%)"%(npad,i+1,npad,n,100 * float(i+1)/n)
    end  = '\r' if i<n-1 else '\n'
    print(strg, end=end, flush=True)


def number_of_electrons_ao(basis, atoms):
    def nel_contrib(a):
        # norm = (2.0*a/np.pi)^3/4
        # integral = (pi/a)^3/2
        return pow(2.0*np.pi/a, 0.75)
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


def nao_for_mol(atoms, lmax, nmax):
    nao = 0
    for q in atoms:
        for l in range(lmax[q]+1):
            nao += (2*l+1)*nmax[(q,l)]
    return nao


def get_training_set(filename, fraction=1.0, sort=True):
    train_selection = np.loadtxt(filename, dtype=int, ndmin=1)
    n = int(fraction*len(train_selection))
    train_configs = train_selection[0:n]
    if sort:
        train_configs.sort()
    return n, train_configs

def get_training_sets(filename, fractions):
    train_selection = np.loadtxt(filename, dtype=int, ndmin=1)
    n = (fractions*len(train_selection)).astype(int)
    train_configs = train_selection[0:n[-1]]
    return len(n), n, train_configs


def get_test_set(filename, nmol):
    train_selection = np.loadtxt(filename, dtype=int)
    test_configs = np.setdiff1d(range(nmol),train_selection)
    return len(test_configs), test_configs


def do_fps(x, d=0):
    # Code from Giulio Imbalzano
    n = len(x)
    if d==0:
        d = n
    iy = np.zeros(d,int)
    measure = np.zeros(d-1,float)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum(x*x, axis=1)
    dl = n2 + n2[iy[0]] - 2.0*np.dot(x,x[iy[0]])
    for i in range(1,d):
        iy[i], measure[i-1] = np.argmax(dl), np.amax(dl)
        nd = n2 + n2[iy[i]] - 2.0*np.dot(x,x[iy[i]])
        dl = np.minimum(dl,nd)
    return iy, measure


def get_atomicindx_new(elements, atomic_numbers):
    element_indices = []
    for imol, atoms in enumerate(atomic_numbers):
        element_indices.append(np.full(len(atoms), -1, int))
        for iq, q in enumerate(elements):
            idx = np.where(atoms==q)
            element_indices[imol][idx] = iq
    return element_indices
