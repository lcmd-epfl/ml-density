import warnings
import numpy as np
import pandas as pd
import ase
import ase.io
from ase.data import chemical_symbols
from pyscf import gto
from qstack.tools import slice_generator
from qstack import compound


def moldata_read(xyzfilename):
    mols = ase.io.read(xyzfilename, ":")
    atomic_numbers = [mol.get_atomic_numbers() for mol in mols]
    return np.array(atomic_numbers, dtype=object)


def get_elements(mols, return_counts=False):
    if len(mols) and isinstance(mols[0], ase.Atoms):
        mols = [mol.numbers for mol in mols]
    return np.unique(np.concatenate(mols), return_counts=return_counts)


class Subset:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.train = self.df[self.df['subset']=='train']['mol_idx'].to_numpy()
        self.test  = self.df[self.df['subset']=='test']['mol_idx'].to_numpy()

    def get_training(self, fraction, sort=True):
        train = self.train[0:int(fraction*len(self.train))]
        return sorted(train) if sort else train

    def get_test(self, sort=True):
        return sorted(self.test) if sort else self.test

    def get_training_all(self, fractions):
        sizes = (fractions*len(self.train)).astype(int)
        train = self.train[0:sizes[-1]]
        return sizes, train


class Basis:

    def __init__(self, basisname, elements):
        self.basisname = basisname
        if isinstance(elements, type({}.keys())):
            elements = list(elements)
        self.elements = np.unique(np.asarray(elements))

        self.lmax, self.nmax, self.ao = {}, {}, {}
        for q in self.elements.tolist():
            self.lmax[q], self.nmax[q], self.ao[q] = self.get_atom_info(q)

        self.msize = np.array([2*l+1 for l in range(max(self.lmax.values())+1)])
        self.nao_atom = {q: self.nmax[q] @ self.msize[:l+1] for q, l in self.lmax.items()}
        self.llist = {q: np.hstack([[l] * n for l, n in enumerate(nmax_q)]).tolist() for q, nmax_q in self.nmax.items()}

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

    def get_atom_info(self, q):
        atom = compound.make_atom(chemical_symbols[q], basis=self.basisname)
        _, l, _ = compound.basis_flatten(atom, return_both=False)
        if not np.all(sorted(l)==l):
            msg = f"Basis functions for {q} are not sorted by angular momentum. This can lead to AO mismatch"
            raise ValueError(msg)
        lmax = l[-1]
        nmax = np.zeros(lmax+1, dtype=int)
        n = []
        m = []
        for li, nao_l in zip(*np.unique(l, return_counts=True), strict=True):
            msize = 2*li+1
            nmax[li] = nao_l//msize
            n.append( np.repeat(np.arange(nmax[li]), msize))
            m.append( np.tile(np.arange(msize)-li, nmax[li]))  # cannot use m from basis_flatten because of pyscf ordering
        ao = np.vstack((np.ones_like(l)*q, l, np.hstack(n), np.hstack(m))).T
        return lmax, nmax, ao

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
        column_order = [iat, q, l, n, m]
        conditions = [self.ao[:,i]==query for i, query in enumerate(column_order) if query is not None]

        if len(conditions)==0:
            return np.arange(len(self.ao))
        else:
            return np.where(np.prod(conditions, axis=0))[0]

    def __repr__(self):
        with np.printoptions(legacy="1.25"):
            return f'AOIndex({self.atoms}, {self.basis})'


def warn_short(*kargs, stacklevel=1, **kwargs):
    def short_warning_formatter(message, category, filename, lineno, line=None):  # noqa ARG001
        return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
    warnings.formatwarning, formatwarning = short_warning_formatter, warnings.formatwarning
    warnings.warn(*kargs, stacklevel=stacklevel+1, **kwargs)
    warnings.formatwarning = formatwarning


def make_pyscf_mol(numbers, positions, basis, spin=None, charge=None, ignore=False):
    mol = gto.Mole()
    mol.atom = [*zip(numbers, positions, strict=True)]
    mol.basis = basis
    if ignore:
        mol.spin = 0
        mol.charge = -(sum(numbers)%2)
    else:
        if spin is not None:
            mol.spin = spin
        if charge is not None:
            mol.charge = charge
    mol.build()
    return mol


def make_dummy_mol(atoms, basis, **kwargs):
    return make_pyscf_mol(numbers=atoms, positions=np.zeros((len(atoms), 3)), basis=basis, **kwargs)
