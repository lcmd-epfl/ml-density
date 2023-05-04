import numpy as np
import equistore
from libs.tmap import vector2tmap, tmap2vector


def print_batches(nfrac, ntrains, paths):
    for i in range(nfrac):
       print(f'batch {i:2d} [{ntrains[i-1]}--{ntrains[i]}):\t {paths[i]}')
    print(flush=True)


def do_work_a(conf, ref_elem, path_proj, path_kern, Avec):
    proj = equistore.load(f'{path_proj}{conf}.npz')
    k_NM = equistore.load(f'{path_kern}{conf}.dat.npz')
    for (l1, q1), pblock in proj:
        msize1 = 2*l1+1
        kblock = k_NM.block(spherical_harmonics_l=l1, species_center=q1)
        ablock = Avec.block(spherical_harmonics_l=l1, species_center=q1)
        for iiref1 in range(np.count_nonzero(ref_elem==q1)):
            dA = np.einsum('kmM,kmn->Mn', kblock.values[:,:,:,iiref1], pblock.values)
            ablock.values[iiref1,:,:] += dA


def get_a(lmax, nmax,
          totsize, ref_elem,
          nfrac, ntrains, trrange,
          path_proj, path_kern, paths_avec):

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
        equistore.save(paths_avec[ifrac]+'.npz', A1)
