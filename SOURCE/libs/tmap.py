from types import SimpleNamespace
import gc
import numpy as np
import equistore

vector_label_names = SimpleNamespace(
    tm = ['spherical_harmonics_l', 'species_center'],
    block_prop = ['radial_channel'],
    block_samp = ['atom_id'],
    block_comp = ['spherical_harmonics_m']
    )

matrix_label_names = SimpleNamespace(
    tm = ['spherical_harmonics_l1', 'spherical_harmonics_l2', 'species_center1', 'species_center2'],
    block_prop = ['radial_channel1', 'radial_channel2'],
    block_samp = ['atom_id1', 'atom_id2'],
    block_comp = ['spherical_harmonics_m1', 'spherical_harmonics_m2']
    )

_molid_name = 'mol_id'

def averages2tmap(averages):
    tm_label_vals = []
    tensor_blocks = []
    for q in averages.keys():
        tm_label_vals.append((0,q))
        values = averages[q].reshape(1,1,-1)
        prop_label_vals = np.arange(values.shape[-1]).reshape(-1,1)
        samp_label_vals = np.array([[0]])
        comp_label_vals = np.array([[0]])
        properties = equistore.Labels(vector_label_names.block_prop, prop_label_vals)
        samples    = equistore.Labels(vector_label_names.block_samp, samp_label_vals)
        components = [equistore.Labels(vector_label_names.block_comp, comp_label_vals)]
        tensor_blocks.append(equistore.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = equistore.Labels(vector_label_names.tm, np.array(tm_label_vals))
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor


def kernels2tmap(atom_charges, kernel):
    tm_label_vals = sorted(list(kernel.keys()), key=lambda x: x[::-1])
    tensor_blocks = []
    for (l, q) in tm_label_vals:
        values = np.ascontiguousarray(np.array(kernel[(l, q)]).transpose(0,2,3,1))
        prop_label_vals = np.arange(values.shape[-1]).reshape(-1,1)
        samp_label_vals = np.where(atom_charges==q)[0].reshape(-1,1)
        comp_label_vals = np.arange(-l, l+1).reshape(-1,1)
        properties = equistore.Labels(vector_label_names.block_prop, prop_label_vals)
        samples    = equistore.Labels(vector_label_names.block_samp, samp_label_vals)
        components = [equistore.Labels([name], comp_label_vals) for name in matrix_label_names.block_comp]
        tensor_blocks.append(equistore.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = equistore.Labels(vector_label_names.tm, np.array(tm_label_vals))
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)
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

    tm_labels = equistore.Labels(vector_label_names.tm, np.array(tm_label_vals))

    block_comp_labels = {key: equistore.Labels(vector_label_names.block_comp, block_comp_label_vals[key]) for key in blocks}
    block_prop_labels = {key: equistore.Labels(vector_label_names.block_prop, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: equistore.Labels(vector_label_names.block_samp, block_samp_label_vals[key]) for key in blocks}

    # Fill in the blocks

    iq = {q:0 for q in elements}
    i = 0
    for iat, q in enumerate(atom_charges):
        for l in range(lmax[q]+1):
            msize = 2*l+1
            nsize = blocks[(l,q)].shape[-1]
            cslice = c[i:i+nsize*msize].reshape(nsize,msize).T
            blocks[(l,q)][iq[q],:,:] = cslice
            i += msize*nsize
        iq[q] += 1
    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)
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
                block = tensor.block(spherical_harmonics_l=l, species_center=q)
                id_samp = block.samples.position((iat,))
                id_prop = block.properties.position((n,))
                for m in range(-l,l+1):
                    id_comp = block.components[0].position((m,))
                    c[i] = block.values[id_samp,id_comp,id_prop]
                    i += 1
    return c


def matrix2tmap(atom_charges, lmax, nmax, dm):

    def pairs(list1, list2):
        return np.array([(i,j) for i in list1 for j in list2])

    elements = sorted(set(atom_charges))

    tm_label_vals = []
    block_prop_label_vals = {}
    block_samp_label_vals = {}
    block_comp_label_vals = {}

    blocks = {}

    # Create labels for TensorMap, lables for blocks, and empty blocks

    for q1 in elements:
        for q2 in elements:
            for l1 in range(lmax[q1]+1):
                for l2 in range(lmax[q2]+1):
                    label = (l1, l2, q1, q2)
                    tm_label_vals.append(label)

                    samples_count1    = np.count_nonzero(atom_charges==q1)
                    components_count1 = 2*l1+1
                    properties_count1 = nmax[(q1,l1)]

                    samples_count2    = np.count_nonzero(atom_charges==q2)
                    components_count2 = 2*l2+1
                    properties_count2 = nmax[(q2,l2)]

                    blocks[label] = np.zeros((samples_count1*samples_count2, components_count1, components_count2, properties_count1*properties_count2))
                    block_comp_label_vals[label] = (np.arange(-l1, l1+1).reshape(-1,1), np.arange(-l2, l2+1).reshape(-1,1))
                    block_prop_label_vals[label] = pairs(np.arange(properties_count1), np.arange(properties_count2))
                    block_samp_label_vals[label] = pairs(np.where(atom_charges==q1)[0],np.where(atom_charges==q2)[0])

    tm_labels = equistore.Labels(matrix_label_names.tm, np.array(tm_label_vals))

    block_prop_labels = {key: equistore.Labels(matrix_label_names.block_prop, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: equistore.Labels(matrix_label_names.block_samp, block_samp_label_vals[key]) for key in blocks}
    block_comp_labels = {key: [equistore.Labels([name], vals) for name, vals in zip(matrix_label_names.block_comp, block_comp_label_vals[key])] for key in blocks}

    # Build tensor blocks
    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]

    # Fill in the blocks

    iq1 = {q1: 0 for q1 in elements}
    i1 = 0
    for iat1, q1 in enumerate(atom_charges):
        for l1 in range(lmax[q1]+1):
            msize1 = 2*l1+1
            nsize1 = nmax[(q1,l1)]
            iq2 = {q2: 0 for q2 in elements}
            i2 = 0
            for iat2, q2 in enumerate(atom_charges):
                for l2 in range(lmax[q2]+1):
                    msize2 = 2*l2+1
                    nsize2 = nmax[(q2,l2)]
                    dmslice = dm[i1:i1+nsize1*msize1,i2:i2+nsize2*msize2].reshape(nsize1,msize1,nsize2,msize2)
                    dmslice = np.transpose(dmslice, axes=[1,3,0,2]).reshape(msize1,msize2,-1)
                    block = tensor_blocks[tm_label_vals.index((l1,l2,q1,q2))]
                    at_p = block.samples.position((iat1,iat2))
                    blocks[(l1,l2,q1,q2)][at_p,:,:,:] = dmslice
                    i2 += msize2*nsize2
                iq2[q2] += 1
            i1 += msize1*nsize1
        iq1[q1] += 1

    # Build tensor map
    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def sparseindices_fill(lmax, nmax, atoms):
    idx = np.zeros((len(atoms), max(lmax.values())+1), dtype=int)
    i = 0
    for iat, q in enumerate(atoms):
        for l in range(lmax[q]+1):
            idx[iat,l] = i
            i += (2*l+1) * nmax[(q,l)]
    return idx


def tmap2matrix(atom_charges, lmax, nmax, tensor):
    nao = int(round(np.sqrt(_get_tsize(tensor))))
    dm = np.zeros((nao, nao))
    idx = sparseindices_fill(lmax, nmax, atom_charges)
    for (l1, l2, q1, q2), block in tensor:
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
        tensor = equistore.load(f'{splitpsfilebase}_{mol_id}.npz')

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


def join(tensors):

    if not all(tensor.keys.names==tensors[0].keys.names for tensor in tensors):
        raise Exception(f'Cannot merge tensors with different label names')
    tm_label_vals = sorted(list(set().union(*[set(tensor.keys.tolist()) for tensor in tensors])))
    tm_labels = equistore.Labels(tensors[0].keys.names, np.array(tm_label_vals))

    blocks = {}
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_labels = {}
    block_samp_label_vals = {}

    for label in tm_labels:
        key = tuple(label.tolist())
        blocks[key] = []
        block_samp_label_vals[key] = []
        for imol,tensor in enumerate(tensors):
            if not label in tensor.keys:
                continue
            block = tensor.block(label)
            blocks[key].append(block.values)
            block_samp_label_vals[key].extend([(imol, *s) for s in block.samples])
            if not key in block_comp_labels:
                block_comp_labels[key] = block.components
                block_prop_labels[key] = block.properties

    for key in blocks:
        blocks[key] = np.concatenate(blocks[key])
        block_samp_label_vals[key] = np.array(block_samp_label_vals[key])
        block_samp_labels[key] = equistore.Labels((_molid_name, *tensor.sample_names), block_samp_label_vals[key])

    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def split(tensor):

    if tensor.sample_names[0]!=_molid_name:
        raise Exception(f'Tensor does not seem to contain several molecules')

    # Check if the molecule indices are continuous
    mollist = sorted(set(np.hstack([np.array(tensor.block(keys).samples.tolist())[:,0] for keys in tensor.keys])))
    if mollist==list(range(len(mollist))):
        tensors = [None] * len(mollist)
    else:
        tensors = {}

    # Common labels
    block_comp_labels = {}
    block_prop_labels = {}
    for label in tensor.keys:
        key = label.tolist()
        block = tensor.block(label)
        block_comp_labels[key] = block.components
        block_prop_labels[key] = block.properties

    # Tensors for each molecule
    for imol in mollist:
        blocks = {}
        block_samp_labels = {}

        for label in tensor.keys:
            key = label.tolist()
            block = tensor.block(label)

            samplelbl = [lbl for lbl in block.samples.tolist() if lbl[0]==imol]
            if len(samplelbl)==0:
                continue
            sampleidx = [block.samples.position(lbl) for lbl in samplelbl]

            blocks[key] = block.values[sampleidx]
            block_samp_labels[key] = equistore.Labels(tensor.sample_names[1:], np.array(samplelbl)[:,1:])

        tm_label_vals = sorted(list(blocks.keys()))
        tm_labels = equistore.Labels(tensor.keys.names, np.array(tm_label_vals))
        tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
        tensors[imol] = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensors


def sph2vector(atoms, lmax, nmax, tensor):
    c = []
    for iat, q in enumerate(atoms):
        c.append(np.squeeze(tensor.block(spherical_harmonics_l=0, species_center=q).values))
        for l in range(1, lmax[q]+1):
            c.append(np.zeros((2*l+1)*nmax[(q,l)]))
    return np.hstack(c)


def tmap_add(x, dx):
    for (l, q) in set(x.keys).intersection(set(dx.keys)):
        b = x.block(spherical_harmonics_l=l, species_center=q)
        db = dx.block(spherical_harmonics_l=l, species_center=q)
        b.values[...] += db.values


def kmm2tmap(qsamples, kernel):
    tm_label_vals = sorted(list(kernel.keys()), key=lambda x: x[::-1])
    tensor_blocks = []
    for (l, q) in tm_label_vals:
        values = kernel[(l, q)].reshape(-1, 2*l+1, 2*l+1, 1)
        prop_label_vals = np.array(1, ndmin=2)
        samp_label_vals = np.array([(*i, *j) for i in qsamples[q] for j in qsamples[q]])
        comp_label_vals = np.arange(-l, l+1).reshape(-1,1)
        properties = equistore.Labels(vector_label_names.block_prop, prop_label_vals)
        samples    = equistore.Labels(('ref_env1', 'ref_env2'), samp_label_vals)
        components = [equistore.Labels([name], comp_label_vals) for name in matrix_label_names.block_comp]
        tensor_blocks.append(equistore.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = equistore.Labels(vector_label_names.tm, np.array(tm_label_vals))
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor
