"""Convert data to TensorMap and back."""

import gc
import numpy as np
import metatensor
from qstack.io import metatensor as equio
from qstack.io.metatensor import _vector_to_tensormap as vector2tmap  # noqa: F401
from qstack.io.metatensor import _tensormap_to_vector as tmap2vector  # noqa: F401


def averages2tmap(averages):
    """Convert per-element spherical averages to TensorMap format.

    Built directly (not via vector2tmap/_vector_to_tensormap) because `averages` only holds the
    l=0 subset of each element's basis, while _vector_to_tensormap derives its angular-momentum
    list from a full pyscf Mole's basis and would require every shell, not just l=0.

    Args:
        averages (dict[int, np.ndarray]): Per-element l=0 average coefficient vectors.

    Returns:
        metatensor.TensorMap: TensorMap storing per-element spherical averages.
    """
    atoms = np.array(sorted(averages.keys()))
    tm_label_vals = [(0, q) for q in atoms]
    tensor_blocks = []
    for iat, q in enumerate(atoms):
        v = averages[q]
        properties = metatensor.Labels(equio.vector_label_names.block_prop, np.arange(len(v)).reshape(-1,1))
        samples    = metatensor.Labels(equio.vector_label_names.block_samp, np.array([[iat]]))
        components = [metatensor.Labels([name], np.array([[0]])) for name in equio.vector_label_names.block_comp]
        tensor_blocks.append(metatensor.TensorBlock(values=v.reshape(1,1,-1), samples=samples, components=components, properties=properties))
    tm_labels = metatensor.Labels(equio.vector_label_names.tm, np.array(tm_label_vals))
    return metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)


def kernels2tmap(atom_charges, kernel):
    """Convert kernel dictionary blocks to TensorMap format.

    Args:
        atom_charges (np.ndarray): Atomic numbers of the query molecule.
        kernel (dict[tuple[int, int], np.4darray[float]]): Kernel blocks keyed by (l, center_type).

    Returns:
        metatensor.TensorMap: Kernel TensorMap with metatensor-compliant labels.
    """
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
    return metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)


def merge_ref_ps(lmax, idx, ps_path_template):
    """Merge per-molecule reference power spectra into one TensorMap.

    Args:
        lmax (dict[int, int]): Maximum angular channel for each center type.
        idx (np.ndarray): Reference rows as (q, mol_id, atom_id).
        ps_path_template (str): Template path to per-molecule power-spectrum files.

    Returns:
        metatensor.TensorMap: Merged reference power-spectrum TensorMap.
    """
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
    return metatensor.TensorMap(keys=tm_labels, blocks=[blocks[key] for key in keys])


def tmap_add(x, dx):
    """Add matching TensorMap blocks from dx into x in place.

    Args:
        x (metatensor.TensorMap): Accumulator TensorMap updated in place.
        dx (metatensor.TensorMap): TensorMap containing increments.
    """
    def keys2set(keys):
        """Convert key labels into a hashable tuple set.

        Args:
            keys (metatensor.Labels): TensorMap key labels.

        Returns:
            set[tuple]: Set of tuple-encoded keys.
        """
        return {tuple(i) for i in keys}

    for (l, q) in keys2set(x.keys).intersection(keys2set(dx.keys)):
        b = x.block(o3_lambda=l, center_type=q)
        db = dx.block(o3_lambda=l, center_type=q)
        b.values[...] += db.values


def kmm2tmap(qsamples, kernel):
    """Convert reference-reference kernel arrays to TensorMap format.

    Args:
        qsamples (dict[int, list[tuple]]): Reference sample labels grouped by center type.
        kernel (dict[tuple[int, int], np.ndarray]): Kernel arrays keyed by (l, center_type).

    Returns:
        metatensor.TensorMap: Reference-reference kernel TensorMap.
    """
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
    return metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)


def read_ps_1mol_l0(psfilename, atomic_numbers):
    """Load l=0 power-spectrum features and reorder rows by atom index.

    Args:
        psfilename (str): Path to one-molecule power-spectrum file.
        atomic_numbers (np.ndarray): Atomic numbers of the molecule.

    Returns:
        np.ndarray: Array of l=0 features ordered by atom index.
    """
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
