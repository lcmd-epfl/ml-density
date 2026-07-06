import logging
import numpy as np
import metatensor
from libs.tmap import vector2tmap, tmap2vector

logger = logging.getLogger('__main__')


def print_batches(ntrains, paths):
    msg = '\n'.join([f'batch {i:2d} [{ntrains[i-1]}--{ntrains[i]}):\t {path}' for i, path in enumerate(paths)])
    logger.info(msg, extra={'flush': True})


def do_work_a(conf, ref_elem, path_proj, path_kern, Avec):
    proj = metatensor.load(path_proj.format(conf))
    k_NM = metatensor.load(path_kern.format(conf))
    for (l1, q1), pblock in proj.items():
        kblock = k_NM.block(o3_lambda=l1, center_type=q1)
        ablock = Avec.block(o3_lambda=l1, center_type=q1)
        for iiref1 in range(np.count_nonzero(ref_elem==q1)):
            dA = np.einsum('kmM,kmn->Mn', kblock.values[:,:,:,iiref1], pblock.values)
            ablock.values[iiref1,:,:] += dA


def get_a(basis, ref_elem, ntrains, trrange, path_proj, path_kern, paths_avec):

    print_batches(ntrains, paths_avec)

    totsize = basis.nao_for_mol(ref_elem)
    Avec = np.zeros(totsize)
    A1 = vector2tmap(ref_elem, basis, Avec)
    for ifrac, path_avec in enumerate(paths_avec):
        for imol in range(ntrains[ifrac-1], ntrains[ifrac]):
            logger.info(f'{0:4d}: {imol:4d}', extra={'flush': True})
            do_work_a(trrange[imol], ref_elem, path_proj, path_kern, A1)
        Avec = tmap2vector(ref_elem, basis, A1)
        np.savetxt(path_avec, Avec)
