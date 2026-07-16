"""Utility functions."""

from typing import NamedTuple
import numpy as np
import pandas as pd
import ase
import ase.io
from ase.data import chemical_symbols
from pyscf import gto
from qstack.tools import slice_generator
from qstack import compound


def moldata_read(xyzfilename):
    """Read all molecules from an XYZ file and return their atomic numbers.

    Args:
        xyzfilename (str): Path to an XYZ file containing one or more molecules.

    Returns:
        np.ndarray: Object array where each item is a 1D atomic-number array.
    """
    mols = ase.io.read(xyzfilename, ":")
    atomic_numbers = [mol.get_atomic_numbers() for mol in mols]
    return np.array(atomic_numbers, dtype=object)


def get_elements(mols, *, return_counts=False):
    """Get unique atomic numbers present in the provided molecules.

    Args:
        mols (list[ase.Atoms] | list[np.ndarray]): Molecules as ASE objects or atomic-number arrays.
        return_counts (bool): Whether to return occurrence counts.

    Returns:
        np.ndarray[int] | dict[int, int]: Unique elements, optionally with counts.
    """
    if len(mols) and isinstance(mols[0], ase.Atoms):
        mols = [mol.numbers for mol in mols]
    elements = np.unique(np.concatenate(mols), return_counts=return_counts)
    return dict(zip(*elements, strict=True)) if return_counts else elements


class Subset:
    """Utility for accessing train/test molecule index subsets."""

    def __init__(self, filename):
        """Load train/test molecule indices from a CSV file.

        The order of the train indices is important and determines
        fractional traning sets.

        Args:
            filename (str): CSV file with columns "subset" and "mol_idx".
        """
        self.df = pd.read_csv(filename)
        self.train = self.df[self.df['subset']=='train']['mol_idx'].to_numpy()
        self.test  = self.df[self.df['subset']=='test']['mol_idx'].to_numpy()

    def get_training(self, fraction=1.0, *, sort=True):
        """Return the first training set indices corresponding to a fraction.

        Fraction 1.0 means the full training set.

        Args:
            fraction (float): Fraction of training indices.
            sort (bool): Whether to return indices sorted increasingly.

        Returns:
            np.ndarray: Training molecule indices for the requested fraction.
        """
        train = self.train[0:int(fraction*len(self.train))]
        return np.sort(train) if sort else train

    def get_test(self, *, sort=True):
        """Return the test set molecule indices.

        Args:
            sort (bool): Whether to return indices sorted increasingly.

        Returns:
            np.ndarray: Test molecule indices.
        """
        return np.sort(self.test) if sort else self.test

    def get_training_all(self, fractions):
        """Return training sizes and the largest corresponding subset for training set fractions.

        Args:
            fractions (np.ndarray): Training fractions.

        Returns:
            tuple[np.ndarray, np.ndarray]: Subset sizes and training indices up to the largest size.
        """
        sizes = (fractions*len(self.train)).astype(int)
        train = self.train[0:sizes[-1]]
        return sizes, train

    def get_pred_idx(self, training, pred_template, frac):
        """Get molecule indices and path to prediction for a subset.

        training (bool): Whether use the training set instead of test.
        pred_template (str): Template path for prediction file.
        frac (float): Training set fraction.

        Returns:
            tuple[np.nddarray[int], str]: Indices of the current set and path to the prediction file.
        """
        if training:
            pred_configs = self.get_training(frac)
            pred_path = pred_template.format(subset='training', train_frac=frac)
        else:
            pred_configs = self.get_test()
            pred_path = pred_template.format(subset='test', train_frac=frac)
        return pred_configs, pred_path


class Basis:
    """Basis metadata and indexing utilities."""

    def __init__(self, basisname, elements):
        """Initialize a Basis instance.

        Args:
            basisname (str): Basis set name understood by PySCF/Q-stack.
            elements (np.ndarray | list[int] | dict_keys[int]): Element / atomic numbers present in the dataset.
        """
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
            return f'{self.__class__.__qualname__}("{self.basisname}", {self.elements})'

    def cat(self, atoms):
        """Build per-AO metadata table for a molecule.

        See also `get_atom_info()`.

        Args:
            atoms (np.ndarray | list[int]): Atomic numbers of the molecule.

        Returns:
            np.ndarray[int]: Array with AO rows [atom_id, q, l, n, m].
        """
        limits = list(slice_generator(atoms, inc=lambda q: len(self.ao[q])))
        ao = np.zeros((limits[-1][1].stop, 5), dtype=int)
        for iat, (q, i) in enumerate(limits):
            ao[i,0] = iat
            ao[i,1:] = self.ao[q]
        return ao

    def get_atom_info(self, q):
        """Return atomic orbitals information.

        The atom info is a tuple (lmax, nmax, ao), where
            lmax (int) : max. angular momentum
            nmax (np.ndarray[int]): number of radial chanels for each angular momentum
            ao (np.ndarray[int]): 2D array containing (q, l, n, m) for each atomic orbital:
                q: atom number
                l: angular momentum
                n: radial channel
                m: magnetic quantum number

        Args:
            q (int): Atom number.

        Returns:
            tuple: Atomic orbitals information.

        Raises:
            ValueError: Basis functions for the element are not sorted by angular momentum.
                        Should never happen for pyscf > 2.X.
        """
        atom = compound.make_atom(chemical_symbols[q], basis=self.basisname)
        _, l, _ = compound.basis_flatten(atom, return_both=False)
        if not np.all(sorted(l)==l):
            msg = f"Basis functions for {q} are not sorted by angular momentum. This can lead to AO mismatch"
            raise ValueError(msg)
        lmax = l[-1].item()
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
        """Compute start indices of each (atom, angular momentum) AO block.

        Args:
            atoms (np.ndarray | list[int]): Atomic numbers of the molecule.

        Returns:
            np.ndarray[int]: Index table with shape (natoms, lmax+1).
        """
        idx = np.zeros((len(atoms), max(self.lmax.values())+1), dtype=int)
        i = 0
        for iat, q in enumerate(atoms):
            for l in range(self.lmax[q]+1):
                idx[iat,l] = i
                i += (2*l+1) * self.nmax[q][l]
        return idx

    def index(self, atoms):
        """Create an AOIndex helper for one molecule.

        Args:
            atoms (np.ndarray | list[int]): Atomic numbers of the molecule.

        Returns:
            AOIndex: Index helper exposing AO metadata queries.
        """
        return AOIndex(atoms, self)

    def nao_for_mol(self, atoms):
        """Return the number of atomic orbitals for a molecule.

        Args:
            atoms (np.ndarray | list[int]): Atomic numbers of the molecule.

        Returns:
            int: Total AO count for the molecule in this basis.
        """
        return sum(self.nao_atom[q] for q in atoms)


class AOIndex:
    """Query helper over per-molecule atomic-orbital metadata."""

    def __init__(self, atoms, basis):
        """Initialize a AOIndex instance.

        Args:
            atoms (np.ndarray | list[int]): Atomic numbers for one molecule.
            basis (Basis): Basis object.
        """
        self.basis = basis
        self.atoms = np.asarray(atoms)
        self.ao = basis.cat(self.atoms)
        self.nao = len(self.ao)
        self.nat = len(self.atoms)

    def find(self, *, iat=None, q=None, l=None, n=None, m=None):
        """Find AO row indices matching the given metadata filters.

        Args:
            iat (int | None): Optional atom index filter.
            q (int | None): Optional atomic-number filter.
            l (int | None): Optional angular momentum filter.
            n (int | None): Optional radial channel filter.
            m (int | None): Optional magnetic quantum number filter.

        Returns:
            np.ndarray[int]: Indices of AO rows satisfying all specified filters.
        """
        column_order = [iat, q, l, n, m]
        conditions = [self.ao[:,i]==query for i, query in enumerate(column_order) if query is not None]
        return np.where(np.prod(conditions, axis=0))[0] if conditions else np.arange(len(self.ao))

    def __repr__(self):
        with np.printoptions(legacy="1.25"):
            return f'{self.__class__.__qualname__}({self.atoms}, {self.basis})'


def make_pyscf_mol(numbers, positions, basis, *, spin=None, charge=None, ignore=False):
    """Construct and build a PySCF Mole object.

    Args:
        numbers (np.ndarray | list[int]): Atomic numbers.
        positions (np.ndarray): Atomic coordinates with shape (natoms, 3).
        basis (str): Basis set name.
        spin (int | None): Spin multiplicity.
        charge (int | None): Molecular charge.
        ignore (bool): If True, derive spin/charge automatically for dummy molecules.

    Returns:
        pyscf.gto.Mole: Built PySCF molecule.
    """
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
    """Create a PySCF molecule with zeroed coordinates for AO bookkeeping.

    Args:
        atoms (np.ndarray | list[int]): Atomic numbers.
        basis (str): Basis-set name.
        **kwargs (object): Extra keyword arguments forwarded to make_pyscf_mol.

    Returns:
        pyscf.gto.Mole: Built PySCF molecule with dummy positions.
    """
    return make_pyscf_mol(numbers=atoms, positions=np.zeros((len(atoms), 3)), basis=basis, **kwargs)


class DatasetPaths(NamedTuple):
    """Specific paths and path templates for power spectra and kernel computation."""
    xyz: str
    power: str
    kernel: str


def get_dataset_paths(p, *, extra=False):
    """Collect specific paths and path templates for power spectra and kernel computation.

    Args:
        p (SimpleNamespace): Paths resolved from the configuration file.
        extra (bool): If True, use extrapolation/out-of-sample dataset.

    Returns:
        DatasetPaths: Paths specific for either the "main" (training) or extrapolation/OOS set.
    """
    if extra:
        return DatasetPaths(p.extra_xyzfilename, p.extra_power_spectrum, p.extra_kernel_nm)
    return DatasetPaths(p.xyzfilename, p.power_spectrum, p.kernel_nm)
