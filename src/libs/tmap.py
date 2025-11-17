from types import SimpleNamespace
import gc
import numpy as np
import metatensor

vector_label_names = SimpleNamespace(
    tm = ['o3_lambda', 'center_type'],
    block_prop = ['radial_channel'],
    block_samp = ['atom_id'],
    block_comp = ['spherical_harmonics_m'],
    )

matrix_label_names = SimpleNamespace(
    tm = ['o3_lambda1', 'o3_lambda2', 'center_type1', 'center_type2'],
    block_prop = ['radial_channel1', 'radial_channel2'],
    block_samp = ['atom_id1', 'atom_id2'],
    block_comp = ['spherical_harmonics_m1', 'spherical_harmonics_m2'],
    )

_molid_name = 'mol_id'


def keys2set(keys):
    return {tuple(i) for i in keys}


def averages2tmap(averages):
    tm_label_vals = []
    tensor_blocks = []
    for q in averages:
        tm_label_vals.append((0,q))
        values = averages[q].reshape(1,1,-1)
        prop_label_vals = np.arange(values.shape[-1]).reshape(-1,1)
        samp_label_vals = np.array([[0]])
        comp_label_vals = np.array([[0]])
        properties = metatensor.Labels(vector_label_names.block_prop, prop_label_vals)
        samples    = metatensor.Labels(vector_label_names.block_samp, samp_label_vals)
        components = [metatensor.Labels(vector_label_names.block_comp, comp_label_vals)]
        tensor_blocks.append(metatensor.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = metatensor.Labels(vector_label_names.tm, np.array(tm_label_vals))
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor


def kernels2tmap(atom_charges, kernel):
    tm_label_vals = sorted(kernel.keys(), key=lambda x: x[::-1])
    tensor_blocks = []
    for (l, q) in tm_label_vals:
        values = np.ascontiguousarray(np.array(kernel[(l, q)]).transpose(0,2,3,1))
        prop_label_vals = np.arange(values.shape[-1]).reshape(-1,1)
        samp_label_vals = np.where(atom_charges==q)[0].reshape(-1,1)
        comp_label_vals = np.arange(-l, l+1).reshape(-1,1)
        properties = metatensor.Labels(vector_label_names.block_prop, prop_label_vals)
        samples    = metatensor.Labels(vector_label_names.block_samp, samp_label_vals)
        components = [metatensor.Labels([name], comp_label_vals) for name in matrix_label_names.block_comp]
        tensor_blocks.append(metatensor.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = metatensor.Labels(vector_label_names.tm, np.array(tm_label_vals))
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor



def vector2tmap(atom_charges, lmax, nmax, c):

    elements = np.unique(atom_charges)

    tm_label_vals = []
    block_prop_label_vals = {}
    block_samp_label_vals = {}
    block_comp_label_vals = {}

    blocks = {}

    # Create labels for TensorMap, lables for blocks, and empty blocks

    for q in elements:
        for l in range(lmax[q]+1):
            label = (l, q)
            tm_label_vals.append(label)
            samples_count    = np.count_nonzero(atom_charges==q)
            components_count = 2*l+1
            properties_count = nmax[(q,l)]
            blocks[label] = np.zeros((samples_count, components_count, properties_count))
            block_comp_label_vals[label] = np.arange(-l, l+1).reshape(-1,1)
            block_prop_label_vals[label] = np.arange(properties_count).reshape(-1,1)
            block_samp_label_vals[label] = np.where(atom_charges==q)[0].reshape(-1,1)

    tm_labels = metatensor.Labels(vector_label_names.tm, np.array(tm_label_vals))

    block_comp_labels = {key: metatensor.Labels(vector_label_names.block_comp, block_comp_label_vals[key]) for key in blocks}
    block_prop_labels = {key: metatensor.Labels(vector_label_names.block_prop, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: metatensor.Labels(vector_label_names.block_samp, block_samp_label_vals[key]) for key in blocks}

    # Fill in the blocks

    iq = dict.fromkeys(elements, 0)
    i = 0
    for q in atom_charges:
        for l in range(lmax[q]+1):
            msize = 2*l+1
            nsize = blocks[(l,q)].shape[-1]
            cslice = c[i:i+nsize*msize].reshape(nsize,msize).T
            blocks[(l,q)][iq[q],:,:] = cslice
            i += msize*nsize
        iq[q] += 1
    tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor


def _get_tsize(tensor):
    return sum([np.prod(tensor.block(key).values.shape) for key in tensor.keys])


def tmap2vector(atom_charges, lmax, nmax, tensor):
    nao = _get_tsize(tensor)
    c = np.zeros(nao)
    i = 0
    for iat, q in enumerate(atom_charges):
        for l in range(lmax[q]+1):
            for n in range(nmax[(q,l)]):
                block = tensor.block(o3_lambda=l, center_type=q)
                id_samp = block.samples.position((iat,))
                id_prop = block.properties.position((n,))
                for m in range(-l,l+1):
                    id_comp = block.components[0].position((m,))
                    c[i] = block.values[id_samp,id_comp,id_prop]
                    i += 1
    return c


def sparseindices_fill(lmax, nmax, atoms):
    idx = np.zeros((len(atoms), max(lmax.values())+1), dtype=int)
    i = 0
    for iat, q in enumerate(atoms):
        for l in range(lmax[q]+1):
            idx[iat,l] = i
            i += (2*l+1) * nmax[(q,l)]
    return idx


def tmap2matrix(atom_charges, lmax, nmax, tensor):
    nao = round(np.sqrt(_get_tsize(tensor)))
    dm = np.zeros((nao, nao))
    idx = sparseindices_fill(lmax, nmax, atom_charges)
    for (l1, l2, q1, q2), block in tensor.items():
        msize1 = 2*l1+1
        msize2 = 2*l2+1
        nsize1 = nmax[(q1,l1)]
        nsize2 = nmax[(q2,l2)]
        ac1 = np.count_nonzero(atom_charges==q1)
        ac2 = np.count_nonzero(atom_charges==q2)
        values = block.values.reshape((ac1, ac2, msize1, msize2, nsize1, nsize2))
        for iiat1, iat1 in enumerate(np.where(atom_charges==q1)[0]):
            for iiat2, iat2 in enumerate(np.where(atom_charges==q2)[0]):
                i1 = idx[iat1,l1]
                i2 = idx[iat2,l2]
                dm[i1:i1+nsize1*msize1,i2:i2+nsize2*msize2] = values[iiat1,iiat2].transpose((2,0,3,1)).reshape((nsize1*msize1,nsize2*msize2))
    return dm


def merge_ref_ps(lmax, elements, atomic_numbers, idx, splitpsfilebase):

    keys = [(l, q) for q in elements for l in range(lmax[q]+1)]

    tm_labels = None
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_label_vals = {key: [] for key in keys}
    blocks = {key: [] for key in keys}

    tensor_keys_names = None
    for iref, (mol_id, atom_id) in enumerate(idx):
        q = atomic_numbers[mol_id][atom_id]
        tensor = metatensor.load(f'{splitpsfilebase}_{mol_id}.mts')

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


def join(tensors):

    if not all(tensor.keys.names==tensors[0].keys.names for tensor in tensors):
        raise ValueError('Cannot merge tensors with different label names')

    tm_label_vals = sorted(set().union(*[keys2set(tensor.keys) for tensor in tensors]))
    tm_labels = metatensor.Labels(tensors[0].keys.names, np.array(tm_label_vals))

    blocks = {}
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_labels = {}
    block_samp_label_vals = {}

    for label in tm_labels:
        key = tuple(label.values.tolist())
        blocks[key] = []
        block_samp_label_vals[key] = []
        for imol,tensor in enumerate(tensors):
            if label not in tensor.keys:
                continue
            block = tensor.block(label)
            blocks[key].append(block.values)
            block_samp_label_vals[key].extend([(imol, *s) for s in block.samples])
            if key not in block_comp_labels:
                block_comp_labels[key] = block.components
                block_prop_labels[key] = block.properties

    for key in blocks:
        blocks[key] = np.concatenate(blocks[key])
        block_samp_label_vals[key] = np.array(block_samp_label_vals[key])
        block_samp_labels[key] = metatensor.Labels((_molid_name, *tensor.sample_names), block_samp_label_vals[key])

    tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def split(tensor):

    if tensor.sample_names[0]!=_molid_name:
        raise ValueError('Tensor does not seem to contain several molecules')

    # Check if the molecule indices are continuous
    mollist = sorted(set(np.hstack([np.array(tensor.block(keys).samples.values.tolist())[:,0] for keys in tensor.keys])))
    if mollist==list(range(len(mollist))):
        tensors = [None] * len(mollist)
    else:
        tensors = {}

    # Common labels
    block_comp_labels = {}
    block_prop_labels = {}
    for label in tensor.keys:
        key = tuple(label.values.tolist())
        block = tensor.block(label)
        block_comp_labels[key] = block.components
        block_prop_labels[key] = block.properties

    # Tensors for each molecule
    for imol in mollist:
        blocks = {}
        block_samp_labels = {}

        for label in tensor.keys:
            key = tuple(label.values.tolist())
            block = tensor.block(label)

            samplelbl = [lbl for lbl in block.samples.values.tolist() if lbl[0]==imol]
            if len(samplelbl)==0:
                continue
            sampleidx = [block.samples.position(lbl) for lbl in samplelbl]

            blocks[key] = block.values[sampleidx]
            block_samp_labels[key] = metatensor.Labels(tensor.sample_names[1:], np.array(samplelbl)[:,1:])

        tm_label_vals = sorted(blocks.keys())
        tm_labels = metatensor.Labels(tensor.keys.names, np.array(tm_label_vals))
        tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
        tensors[imol] = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensors


def sph2vector(atoms, lmax, nmax, tensor):
    c = []
    for q in atoms:
        c.append(np.squeeze(tensor.block(o3_lambda=0, center_type=q).values))
        c.extend([np.zeros((2*l+1)*nmax[(q,l)]) for l in range(1, lmax[q]+1)])
    return np.hstack(c)


def tmap_add(x, dx):
    for (l, q) in keys2set(x.keys).intersection(keys2set(dx.keys)):
        b = x.block(o3_lambda=l, center_type=q)
        db = dx.block(o3_lambda=l, center_type=q)
        b.values[...] += db.values


def kmm2tmap(qsamples, kernel):
    tm_label_vals = sorted(kernel.keys(), key=lambda x: x[::-1])
    tensor_blocks = []
    for (l, q) in tm_label_vals:
        values = kernel[(l, q)].reshape(-1, 2*l+1, 2*l+1, 1)
        prop_label_vals = np.array(1, ndmin=2)
        samp_label_vals = np.array([(*i, *j) for i in qsamples[q] for j in qsamples[q]])
        comp_label_vals = np.arange(-l, l+1).reshape(-1,1)
        properties = metatensor.Labels(vector_label_names.block_prop, prop_label_vals)
        samples    = metatensor.Labels(('ref_env1', 'ref_env2'), samp_label_vals)
        components = [metatensor.Labels([name], comp_label_vals) for name in matrix_label_names.block_comp]
        tensor_blocks.append(metatensor.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = metatensor.Labels(vector_label_names.tm, np.array(tm_label_vals))
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor
