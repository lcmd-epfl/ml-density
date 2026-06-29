import numpy as np
import ase.io


def moldata_read(xyzfilename: str) -> np.ndarray:
    '''Read (possibly concatenated) XYZ file.

    Returns object array of shape (nmol,) where each element is a 1D int array of atomic numbers for that molecule.'''
    mols = ase.io.read(xyzfilename, ":")
    atomic_numbers = []
    for mol in mols:
        atomic_numbers.append(mol.get_atomic_numbers())
    return np.array(atomic_numbers, dtype=object)


def get_elements_list(atomic_numbers: np.ndarray, return_counts: bool = False) -> np.ndarray:
    '''Return sorted unique elements across all molecules.'''
    return np.unique(np.concatenate(atomic_numbers), return_counts=return_counts)


def print_progress(i: int, n: int) -> None:
    '''Print a progress bar for step i out of n.'''
    npad = len(str(n))
    # strg = "Doing point %*i of %*i (%5.1f %%)"%(npad,i+1,npad,n,100 * float(i+1)/n)
    strg = f"Doing point {i+1:{npad}d} of {n:{npad}d} ({100 * (i+1) / n:5.1f} %)"
    end  = '\r' if i<n-1 else '\n'
    print(strg, end=end, flush=True)


def number_of_electrons_ao(basis: dict, atoms: np.ndarray) -> np.ndarray:
    '''Return per-AO electron number contributions from s-type Gaussians.'''
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


def correct_number_of_electrons(c: np.ndarray, S: np.ndarray, q: np.ndarray, N: float) -> np.ndarray:
    '''Project coefficients c so that the integrated density equals N electrons.'''
    S1q  = np.linalg.solve(S, q)
    return c + S1q * (N - c@q)/(q@S1q)


def nao_for_mol(atoms: np.ndarray, lmax: dict, nmax: dict) -> int:
    '''Return total number of auxiliary basis functions for a molecule.'''
    nao = 0
    for q in atoms:
        for l in range(lmax[q]+1):
            nao += (2*l+1)*nmax[(q,l)]
    return nao


def get_training_set(filename: str, fraction: float = 1.0, sort: bool = True) -> tuple[int, np.ndarray]:
    '''Load training indices and return the first fraction of them.'''
    train_selection = np.loadtxt(filename, dtype=int, ndmin=1)
    n = int(fraction*len(train_selection))
    train_configs = train_selection[0:n]
    if sort:
        train_configs.sort()
    return n, train_configs


def get_training_sets(filename: str, fractions: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    '''Load training indices for multiple fractions at once.'''
    train_selection = np.loadtxt(filename, dtype=int, ndmin=1)
    n = (fractions*len(train_selection)).astype(int)
    train_configs = train_selection[0:n[-1]]
    return len(n), n, train_configs


def get_test_set(filename: str, nmol: int) -> tuple[int, np.ndarray]:
    '''Return number of molecules and their indices in the testing set (molecules not in the training set).'''
    train_selection = np.loadtxt(filename, dtype=int)
    test_configs = np.setdiff1d(range(nmol), train_selection)
    return len(test_configs), test_configs


def do_fps(x: np.ndarray, d: int = 0) -> tuple[np.ndarray, np.ndarray]:
    '''Farthest Point Sampling: select d maximally diverse points from x.'''
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
