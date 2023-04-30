import numpy as np

def reorder_ps(power_per_conf, power, nel, atom_counting, atomicindx):
    iat = 0
    for iel in range(nel):
        for icount in range(atom_counting[iel]):
            jat = atomicindx[iel,icount]
            power_per_conf[jat] = power[iat]
            iat+=1
    return


def read_ps(psfilename, nmol, nel, atom_counting, atomicindx):

    power = np.load(psfilename)
    nfeat = power.shape[-1]
    power_per_conf = np.zeros(power.shape,float)

    for imol in range(nmol):
        reorder_ps(power_per_conf[imol], power[imol], nel, atom_counting[imol], atomicindx[imol])
    return (nfeat, power_per_conf)
