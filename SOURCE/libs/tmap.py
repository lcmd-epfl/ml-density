from types import SimpleNamespace
import numpy as np
import equistore

vector_label_names = SimpleNamespace(
    tm = ['spherical_harmonics_l', 'species_center'],
    block_prop = ['radial_channel'],
    block_samp = ['atom_id'],
    block_comp = ['spherical_harmonics_m']
    )

matrix_label_names = SimpleNamespace(
    tm = ['spherical_harmonics_l1', 'spherical_harmonics_l2', 'element1', 'element2'],
    block_prop = ['radial_channel1', 'radial_channel2'],
    block_samp = ['atom_id1', 'atom_id2'],
    block_comp = ['spherical_harmonics_m1', 'spherical_harmonics_m2']
    )


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
