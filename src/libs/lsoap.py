import numpy as np
from featomic import SphericalExpansion
from featomic.clebsch_gordan import EquivariantPowerSpectrum
from metatensor import Labels, operations


MIN_NORM = 1e-10


def ps_normalize_inplace(vals, min_norm=MIN_NORM):
    norm = np.sqrt(np.linalg.norm(vals @ vals.T))
    if norm > min_norm:
        vals /= norm
    else:
        vals[...] = 0.0
    return norm


def ps_normalize_gradient_inplace(idx, grad, values, norm, min_norm=MIN_NORM):
    # print(grad[idx,:,:].shape)  # natoms-in-mol * 3 * (2*l+1) * nfeatures
    # print(values.shape)         # (2*l+1) * nfeatures
    if norm > min_norm:
        if values.shape[0]==1:
            t1 = np.einsum('kxmi,mi->kx', grad[idx], values)
            dnorm = np.einsum('kx,mi->kxmi', t1, values)
        else:
            p = values @ values.T
            p1 = np.einsum('Mf,kxmf->Mmkx', values, grad[idx])
            p2 = p1.transpose(1,0,2,3)
            t1 = np.einsum('Mm,Mmkx->kx', p, p1+p2)
            dnorm = 0.5*np.einsum('kx,mi->kxmi', t1, values)
        grad[idx,...] -= dnorm
        grad[idx,...] /= norm
    else:
        grad[idx,...] = 0.0


def normalize_tensormap(soap, min_norm=MIN_NORM):
    for _key, block in soap.items():  # noqa PERF102
        for samp in block.samples:
            isamp = block.samples.position(samp)
            norm = ps_normalize_inplace(block.values[isamp,:,:], min_norm=min_norm)
            if block.has_gradient('positions'):
                gradient = block.gradient('positions')
                # get indices for gradient of center id #isamp
                igsamps = np.array([i for i, gsamp in enumerate(gradient.samples) if gsamp[0] == isamp])
                ps_normalize_gradient_inplace(igsamps, gradient.values, block.values[isamp,:,:], norm, min_norm=min_norm)


class EquivariantPowerSpectrum_custom(EquivariantPowerSpectrum):

    def __init__(self, calculator_1, calculator_2=None, neighbor_types=None, *, dtype=None, device=None,
                 filter_redundant_keys='all'):

        super().__init__(calculator_1, calculator_2, neighbor_types, dtype=dtype, device=device)

        if filter_redundant_keys == 'all':
            from featomic.clebsch_gordan._density_correlations import _filter_redundant_keys
        elif filter_redundant_keys == 'odd':
            _filter_redundant_keys = self.filter_redundant_keys_odd
        else:
            _filter_redundant_keys = None

        import json
        parameters_1 = json.loads(self.calculator_1.parameters)
        if self.calculator_2 is None:
                parameters_2 = parameters_1
        max_angular = parameters_1['basis']['max_angular'] + parameters_2['basis']['max_angular']

        from featomic.clebsch_gordan._cg_product import ClebschGordanProduct
        self._cg_product = ClebschGordanProduct(
             max_angular=max_angular,
             cg_backend=None,
             keys_filter=_filter_redundant_keys,
             arrays_backend=None,
             dtype=None,
             device=None,
         )

    def filter_redundant_keys_odd(self, keys):
        from featomic.clebsch_gordan import _dispatch
        """
        Filter redundant keys from the ``keys`` to only keep keys where the l values are
        sorted (i.e. l1 <= l2 <= ... <= ln). These are redundant when handling
        auto-correlations of a density.
        Keep also all keys where o3_lambda is even.
        """
        nu_target = 2
        l_list_idxs = [ keys.names.index(f"l_{o3_lambda}") for o3_lambda in range(1, nu_target + 1) ]
        keys_to_keep = []
        for key_idx in range(len(keys)):
            key = keys.entry(key_idx)
            l_list_values = _dispatch.to_int_list(key.values[l_list_idxs])
            is_l_sorted = _dispatch.all(_dispatch.int_array_like(l_list_values, like=key.values) == _dispatch.int_array_like(sorted(l_list_values), like=key.values))
            if is_l_sorted or (key["o3_lambda"]%2==0):
                keys_to_keep.append(key_idx)
        return keys_to_keep


def generate_lambda_soap_wrapper(mols: list, rascal_hypers: dict, neighbor_species=None, normalize=True, min_norm=MIN_NORM, lmax=None, gradients=None):

    if gradients is not None:
        raise NotImplementedError("Gradients are not implemented yet")

    if not isinstance(mols, list):
        mols = [mols]

    spex_calculator = SphericalExpansion(**rascal_hypers)

    # filter redundant keys to make PS smaller
    # but break compatibility with previous results because lead to different kernels with even L>0
    # calculator = EquivariantPowerSpectrum(spex_calculator, neighbor_types=neighbor_species)

    # legacy: do no filter redundant keys
    # calculator = EquivariantPowerSpectrum_custom(spex_calculator, neighbor_types=neighbor_species, filter_redundant_keys=False)

    # new: filter only odd redundant keys so the PS are smaller but still compatible with previous results for even L
    calculator = EquivariantPowerSpectrum_custom(spex_calculator, neighbor_types=neighbor_species, filter_redundant_keys='odd')

    if lmax is None:
        selected_keys = Labels(["o3_sigma"], np.array([[1]]).T)
    else:
        selected_keys = np.vstack([np.pad(np.arange(l+1)[:,None], ((0,0),(1, 1)), constant_values=(1,q)) for q, l in lmax.items()])
        selected_keys = Labels(["o3_sigma", "o3_lambda", "center_type"], selected_keys)

    soap = calculator.compute(mols, neighbors_to_properties=True, selected_keys=selected_keys)
    soap = operations.remove_dimension(soap, "keys", "o3_sigma")

    if normalize:
        normalize_tensormap(soap, min_norm=min_norm)
    return soap


def make_rascal_hypers(soap_rcut, soap_ncut, soap_lcut, soap_sigma):
    return {
           "cutoff": {
               "radius": soap_rcut,
               "smoothing": {
                   "type": "ShiftedCosine",
                   "width": 0.5,
               },
           },
           "density": {
               "type": "Gaussian",
               "width": soap_sigma,
               "center_atom_weight": 1.0,
           },
           "basis": {
               "type": "TensorProduct",
               "max_angular": soap_lcut,
               "radial": {
                   "type": "Gto",
                   "max_radial": soap_ncut,
               },
           },
       }
