import copy
import itertools
import numpy as np
import ase.io
from ase.data import chemical_symbols
import pyscf
from qstack.tools import slice_generator
from qstack import compound


def moldata_read(xyzfilename):
    mols = ase.io.read(xyzfilename, ":")
    atomic_numbers = [mol.get_atomic_numbers() for mol in mols]
    return np.array(atomic_numbers, dtype=object)


def get_elements_list(atomic_numbers, return_counts=False):
    return np.unique(np.concatenate(atomic_numbers), return_counts=return_counts)


def print_progress(i, n):
    npad = len(str(n))
    strg = "Doing point %*i of %*i (%5.1f %%)"%(npad,i+1,npad,n,100 * float(i+1)/n)
    end  = '\r' if i<n-1 else '\n'
    print(strg, end=end, flush=True)


def number_of_electrons_ao(basis, atoms):
    def nel_contrib(a):
        # L2 norm = (pi/(2.0*a))^3/4  (\int \phi^2(\vec r) \de^3 \vec r)^(1/2)
        # L1 int  = (pi/a)^3/2        (\int \phi(\vec r) \de^3 \vec r)
        # nel_contrib = L1/L2
        return pow(2.0*np.pi/a, 0.75)
    def ssover(a1, a2):
        # <G(a1)|G(a2)> = gau3int(a1+a2) / np.sqrt(gau3int(2*a1) * gau3int(2*a2))
        # gau3int(a): np.power(np.pi/a, 1.5)
        return np.power(np.sqrt(a1*a2)/(0.5*(a1+a2)), 1.5)
    def renorm_orbital(a, w):
        aa = np.zeros((len(a), len(a)))
        for i, ai in enumerate(a):
            for j, aj in enumerate(a):
                aa[i,j] = ssover(ai, aj)
        return 1.0/np.sqrt(w @ aa @ w)

    nel = []
    for q in atoms:
        for l, gto in basis[q]:
            if l==0:
                a, w = np.array(gto).T
                nel.append(renorm_orbital(a, w) * w @ nel_contrib(a))
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
    test_configs = np.setdiff1d(range(nmol), train_selection)
    return len(test_configs), test_configs


class Basis:
    #def __init__(self, lmax, nmax):
    #    ao = {}
    #    for q in lmax.keys():
    #        ao[q] = []
    #        for l in range(lmax[q]+1):
    #            for n in range(nmax[(q,l)]):
    #                for m in range(-l, l+1):
    #                    ao[q].append((q, l, n, m))
    #        ao[q] = np.array(ao[q])
    #    self.ao = ao
    #    self.lmax = lmax
    #    self.nmax = nmax
    def __init__(self, basisname, elements):
        lmax = {}
        nmax = {}
        ao = {}
        for q in elements:
            atom = compound.make_atom(chemical_symbols[q], basis=basisname)
            _, l, _ = compound.basis_flatten(atom, return_both=False)
            assert np.all(sorted(l)==l)
            lmax[q] = l[-1]
            n = []
            m = []
            for li, nao_l in zip(*np.unique(l, return_counts=True)):
                msize = 2*li+1
                nmax[q,li] = nao_l//msize
                n.append( np.repeat(np.arange(nmax[q,li]), msize))
                m.append( np.tile(np.arange(msize)-li, nmax[q,li]))  # cannot use m from basis_flatten because of pyscf ordering
            ao[q] = np.vstack((np.ones_like(l)*q, l, np.hstack(n), np.hstack(m))).T
        self.ao = ao
        self.lmax = lmax
        self.nmax = nmax


    def cat(self, atoms):
        limits = list(slice_generator(atoms, inc=lambda q: len(self.ao[q])))
        ao = np.zeros((limits[-1][1].stop, 5), dtype=int)
        for iat, (q, i) in enumerate(limits):
            ao[i,0] = iat
            ao[i,1:] = self.ao[q]
        return ao

    def index(self, atoms):
        return AOIndex(atoms, self)


class AOIndex:
    def __init__(self, atoms, basis):
        self.ao = basis.cat(atoms)
        self.nao = len(self.ao)
        self.nat = len(atoms)
        self.atoms = atoms

    def find(self, iat=None, q=None, l=None, n=None, m=None):
        conditions = []
        if iat is not None:
            conditions.append(self.ao[:,0]==iat)
        if q is not None:
            conditions.append(self.ao[:,1]==q)
        if l is not None:
            conditions.append(self.ao[:,2]==l)
        if n is not None:
            conditions.append(self.ao[:,3]==n)
        if m is not None:
            conditions.append(self.ao[:,4]==m)

        if len(conditions)==0:
            return np.arange(len(self.ao))
        else:
            return np.where(np.prod(conditions, axis=0))[0]
