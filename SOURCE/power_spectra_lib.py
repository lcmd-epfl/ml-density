import gc
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

###################################################

def merge_ref_ps(refs, lmax, elements, atomic_numbers, idx, filetemplate):

    keys = [(l, q) for q in elements for l in range(lmax[q]+1)]

    tm_labels = None
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_label_vals = {key: [] for key in keys}
    blocks = {key: [] for key in keys}

    tensor_keys_names = None
    for iref, ref in enumerate(idx):
        mol_id, atom_id  = ref
        q = atomic_numbers[mol_id][atom_id]
        tensor = equistore.load(eval("f'{}'".format(filetemplate)))

        for l in range(lmax[q]+1):
            key = (l, q)
            block = tensor.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, atom_id))
            vals  = np.copy(block.values[isamp,:,:])
            blocks[key].append(vals)
            block_samp_label_vals[key].append(iref)
            if key not in block_comp_labels:
                block_comp_labels[key] = block.components
                block_prop_labels[key] = block.properties
        if not tensor_keys_names:
            tensor_keys_names = tensor.keys.names

        del tensor
        gc.collect()

    for key in keys:
        block_samp_label = equistore.Labels(['ref_env'], np.array(block_samp_label_vals[key]).reshape(-1,1))
        blocks[key] = equistore.TensorBlock(values=np.array(blocks[key]),
                                            samples=block_samp_label,
                                            components=block_comp_labels[key],
                                            properties=block_prop_labels[key])

    tm_labels = equistore.Labels(tensor_keys_names, np.array(keys))
    tensor = equistore.TensorMap(keys=tm_labels, blocks=[blocks[key] for key in keys])
    return tensor



def get_ref_idx(natoms, refs):
    idx_mol = []
    idx_atm = []
    for imol, nat in enumerate(natoms):
        idx_mol += [imol] * nat
        idx_atm += range(nat)
    return np.array(idx_mol)[refs], np.array(idx_atm)[refs]


