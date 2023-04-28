import numpy as np
import equistore

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

def read_ps_1mol(psfilename, nel, atom_counting, atomicindx):

    power = np.load(psfilename)
    nfeat = power.shape[-1]
    power_per_conf = np.zeros(power.shape[1:],float)
    reorder_ps(power_per_conf, power[0], nel, atom_counting, atomicindx)
    return (nfeat, power_per_conf)


def reorder_ps_new(power_per_conf, elements, atomic_numbers, power):
    i=0
    for q in elements:
        idx = np.where(atomic_numbers == q)
        power_per_conf[idx] = power[i:i+len(idx[0])]
        i+=len(idx[0])
    return

def read_ps_1mol_new1(psfilename, elements, atomic_numbers):
    power = np.squeeze(np.load(psfilename))
    power_per_conf = np.zeros_like(power)
    reorder_ps_new(power_per_conf, elements, atomic_numbers, power)
    return power_per_conf

def read_ps_1mol_l0(psfilename, atomic_numbers):
    power_sorted = None
    power = equistore.load(psfilename)
    for q in set(atomic_numbers):
        idx = np.where(atomic_numbers == q)
        block = power.block(spherical_harmonics_l=0, species_center=q)
        if power_sorted is None:
            power_sorted = np.zeros((len(atomic_numbers), block.values.shape[-1]))
        power_sorted[idx] = block.values[:,0,:]
    return power_sorted
