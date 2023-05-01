from types import SimpleNamespace
import numpy as np
import equistore

vector_label_names = SimpleNamespace(
    tm = ['spherical_harmonics_l', 'species_center'],
    block_prop = ['radial_channel'],
    block_samp = ['atom_id'],
    block_comp = ['spherical_harmonics_m']
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
