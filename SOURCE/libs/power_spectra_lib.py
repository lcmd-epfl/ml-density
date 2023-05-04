import gc
import numpy as np
import equistore


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
        power_sorted[idx] = np.copy(block.values[:,0,:])
    del power
    gc.collect()
    return power_sorted
