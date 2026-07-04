import gc
import numpy as np
import metatensor
from qstack.io import metatensor as equio


def vector2tmap(atom_charges, basis, c):
    return equio._vector_to_tensormap(atom_charges, basis.llist, c)


def tmap2vector(atom_charges, basis, tensor):
    return equio._tensormap_to_vector(atom_charges, basis.llist, tensor)


def tmap2matrix(atom_charges, basis, tensor):
    return equio._tensormap_to_matrix(atom_charges, basis.llist, tensor, fast=True)


def averages2tmap(averages):
    atoms = np.array(sorted(averages.keys()))
    llist = {q:[0]*len(v) for q, v in averages.items()}
    c = np.hstack([averages[q] for q in atoms])
    return equio._vector_to_tensormap(atoms, llist, c)


def kernels2tmap(atom_charges, kernel):
    tm_label_vals = sorted(kernel.keys(), key=lambda x: x[::-1])
    tensor_blocks = []
    for (l, q) in tm_label_vals:
        values = np.ascontiguousarray(np.array(kernel[l,q]).transpose(0,2,3,1))
        prop_label_vals = np.arange(values.shape[-1]).reshape(-1,1)
        samp_label_vals = np.where(atom_charges==q)[0].reshape(-1,1)
        comp_label_vals = np.arange(-l, l+1).reshape(-1,1)
        properties = metatensor.Labels(equio.vector_label_names.block_prop, prop_label_vals)
        samples    = metatensor.Labels(equio.vector_label_names.block_samp, samp_label_vals)
        components = [metatensor.Labels([name], comp_label_vals) for name in equio.matrix_label_names.block_comp]
        tensor_blocks.append(metatensor.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = metatensor.Labels(equio.vector_label_names.tm, np.array(tm_label_vals))
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor


def merge_ref_ps(lmax, idx, ps_path_template):

    keys = [(l, q) for q in sorted(lmax.keys()) for l in range(lmax[q]+1)]

    tm_labels = None
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_label_vals = {key: [] for key in keys}
    blocks = {key: [] for key in keys}

    tensor_keys_names = None
    for iref, (q, mol_id, atom_id) in enumerate(idx):
        tensor = metatensor.load(ps_path_template.format(mol_id))

        for l in range(lmax[q]+1):
            key = (l, q)
            block = tensor.block(o3_lambda=l, center_type=q)
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
        block_samp_label = metatensor.Labels(['ref_env'], np.array(block_samp_label_vals[key]).reshape(-1,1))
        blocks[key] = metatensor.TensorBlock(values=np.array(blocks[key]),
                                             samples=block_samp_label,
                                             components=block_comp_labels[key],
                                             properties=block_prop_labels[key])

    tm_labels = metatensor.Labels(tensor_keys_names, np.array(keys))
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=[blocks[key] for key in keys])
    return tensor


def sph2vector(atoms, basis, tensor):
    return np.hstack([
            np.pad(np.squeeze(tensor.block(o3_lambda=0, center_type=q).values), (0, basis.nao_atom[q]-basis.nmax[q][0]))
           for q in atoms])


def tmap_add(x, dx):

    def keys2set(keys):
        return {tuple(i) for i in keys}

    for (l, q) in keys2set(x.keys).intersection(keys2set(dx.keys)):
        b = x.block(o3_lambda=l, center_type=q)
        db = dx.block(o3_lambda=l, center_type=q)
        b.values[...] += db.values


def kmm2tmap(qsamples, kernel):
    tm_label_vals = sorted(kernel.keys(), key=lambda x: x[::-1])
    tensor_blocks = []
    for (l, q) in tm_label_vals:
        values = kernel[l, q].reshape(-1, 2*l+1, 2*l+1, 1)
        prop_label_vals = np.array(1, ndmin=2)
        samp_label_vals = np.array([(*i, *j) for i in qsamples[q] for j in qsamples[q]])
        comp_label_vals = np.arange(-l, l+1).reshape(-1,1)
        properties = metatensor.Labels(equio.vector_label_names.block_prop, prop_label_vals)
        samples    = metatensor.Labels(('ref_env1', 'ref_env2'), samp_label_vals)
        components = [metatensor.Labels([name], comp_label_vals) for name in equio.matrix_label_names.block_comp]
        tensor_blocks.append(metatensor.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = metatensor.Labels(equio.vector_label_names.tm, np.array(tm_label_vals))
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor


def read_ps_1mol_l0(psfilename, atomic_numbers):
    power_sorted = None
    power = metatensor.load(psfilename)
    for q in set(atomic_numbers):
        idx = np.where(atomic_numbers==q)
        block = power.block(o3_lambda=0, center_type=q)
        if power_sorted is None:
            power_sorted = np.zeros((len(atomic_numbers), block.values.shape[-1]))
        power_sorted[idx] = np.copy(block.values[:,0,:])
    del power
    gc.collect()
    return power_sorted
