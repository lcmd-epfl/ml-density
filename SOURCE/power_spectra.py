import numpy as np

def reorder_ps(power_per_conf, power, nspecies, atom_counting, atomicindx):
    iat = 0
    for ispe in range(nspecies):
        for icount in range(atom_counting[ispe]):
            jat = atomicindx[ispe,icount]
            power_per_conf[jat] = power[iat]
            iat+=1
    return


def read_ps(psfilename, ndata, nspecies, atom_counting, atomicindx):

    power = np.load(psfilename)
    nfeat = power.shape[-1]
    power_per_conf = np.zeros(power.shape,float)

    for iconf in range(ndata):
        reorder_ps(power_per_conf[iconf], power[iconf], nspecies, atom_counting[iconf], atomicindx[iconf])
    return (nfeat, power_per_conf)

def read_ps_1mol(psfilename, nspecies, atom_counting, atomicindx):

    power = np.load(psfilename)
    nfeat = power.shape[-1]
    power_per_conf = np.zeros(power.shape[1:],float)
    reorder_ps(power_per_conf, power[0], nspecies, atom_counting, atomicindx)
    return (nfeat, power_per_conf)

