"""If I understand corretly, A is \mathbf{K}_{NM}^\mathrm{T}\mathbf{w}, with w being the projection of the electron density on the atomic orbitals
"""
import numpy as np
import metatensor
from libs.tmap import vector2tmap, tmap2vector


def print_batches(nfrac: int, ntrains: np.ndarray, paths: list[str]) -> None:
    '''Print batch ranges and output file paths.'''
    for i in range(nfrac):
       print(f'batch {i:2d} [{ntrains[i-1]}--{ntrains[i]}):\t {paths[i]}')
    print(flush=True)


def do_work_a(conf: int, ref_elem: np.ndarray, path_proj: str, path_kern: str, Avec: metatensor.TensorMap) -> None:
    '''Accumulate one molecule's contribution to the target projection vector A.'''
    proj = metatensor.load(f'{path_proj}{conf}.mts')
    k_NM = metatensor.load(f'{path_kern}{conf}.mts')
    for (l1, q1), pblock in proj.items():
        msize1 = 2*l1+1
        kblock = k_NM.block(o3_lambda=l1, center_type=q1)
        ablock = Avec.block(o3_lambda=l1, center_type=q1)
        for iiref1 in range(np.count_nonzero(ref_elem==q1)):
            dA = np.einsum('kmM,kmn->Mn', kblock.values[:,:,:,iiref1], pblock.values)
            ablock.values[iiref1,:,:] += dA


def get_a(lmax: dict, nmax: dict,
          totsize: int, ref_elem: np.ndarray,
          nfrac: int, ntrains: np.ndarray, trrange: np.ndarray,
          path_proj: str, path_kern: str, paths_avec: list[str]) -> None:
    '''Build the target projection A = sum_n K_NM^T w_n and save to disk.'''

    ntrains = np.pad(ntrains, (0, 1), 'constant', constant_values=0)
    print_batches(nfrac, ntrains, paths_avec)

    Avec = np.zeros(totsize)
    A1 = vector2tmap(ref_elem, lmax, nmax, Avec)
    for ifrac in range(nfrac):
        for imol in range(ntrains[ifrac-1], ntrains[ifrac]):
            print(f'{0:4d}: {imol:4d}')
            do_work_a(trrange[imol], ref_elem, path_proj, path_kern, A1)
        Avec = tmap2vector(ref_elem, lmax, nmax, A1)
        np.savetxt(paths_avec[ifrac], Avec)
