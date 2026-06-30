import numpy as np
import ase.io
from ase.data import chemical_symbols
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
    def __init__(self, basisname, elements):
        self.basisname = basisname
        self.elements = sorted(elements)
        lmax = {}
        nmax = {}
        ao = {}
        for q in elements:
            atom = compound.make_atom(chemical_symbols[q], basis=basisname)
            _, l, _ = compound.basis_flatten(atom, return_both=False)
            if not np.all(sorted(l)==l):
                raise ValueError("Basis functions are not sorted by angular momentum")
            lmax[q] = l[-1]
            nmax[q] = np.zeros(lmax[q]+1, dtype=int)
            n = []
            m = []
            for li, nao_l in zip(*np.unique(l, return_counts=True), strict=True):
                msize = 2*li+1
                nmax[q][li] = nao_l//msize
                n.append( np.repeat(np.arange(nmax[q][li]), msize))
                m.append( np.tile(np.arange(msize)-li, nmax[q][li]))  # cannot use m from basis_flatten because of pyscf ordering
            ao[q] = np.vstack((np.ones_like(l)*q, l, np.hstack(n), np.hstack(m))).T
        self.ao = ao
        self.lmax = lmax
        self.nmax = nmax

        self.msize = np.array([2*l+1 for l in range(max(lmax.values())+1)])
        self.nao_atom = {q: nmax[q] @ self.msize[:l+1] for q, l in lmax.items()}

    def __repr__(self):
        with np.printoptions(legacy="1.25"):
            return f'Basis("{self.basisname}", {self.elements})'

    def cat(self, atoms):
        limits = list(slice_generator(atoms, inc=lambda q: len(self.ao[q])))
        ao = np.zeros((limits[-1][1].stop, 5), dtype=int)
        for iat, (q, i) in enumerate(limits):
            ao[i,0] = iat
            ao[i,1:] = self.ao[q]
        return ao

    def sparse_indices(self, atoms):
        idx = np.zeros((len(atoms), max(self.lmax.values())+1), dtype=int)
        i = 0
        for iat, q in enumerate(atoms):
            for l in range(self.lmax[q]+1):
                idx[iat,l] = i
                i += (2*l+1) * self.nmax[q][l]
        return idx

    def index(self, atoms):
        return AOIndex(atoms, self)

    def nao_for_mol(self, atoms):
        return sum(self.nao_atom[q] for q in atoms)


class AOIndex:
    def __init__(self, atoms, basis):
        self.basis = basis
        self.atoms = np.asarray(atoms)
        self.ao = basis.cat(self.atoms)
        self.nao = len(self.ao)
        self.nat = len(self.atoms)

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

    def __repr__(self):
        with np.printoptions(legacy="1.25"):
            return f'AOIndex({self.atoms}, {self.basis})'
