import numpy as np
import equistore


def kernel_nm_sparse_indices(natoms, lmax, ref_elements, el_dict, atom_counting):
    llmax = max(lmax.values())
    kernel_sparse_indices = np.zeros((len(ref_elements), natoms, llmax+1), dtype=int)
    kernel_size = 0
    for iref, iel in enumerate(ref_elements):
        q = el_dict[iel]
        for l in range(lmax[q]+1):
            msize = 2*l+1
            for iat in range(atom_counting[iel]):
                kernel_sparse_indices[iref,iat,l] = kernel_size
                kernel_size += msize*msize
    return kernel_size, kernel_sparse_indices


def kernel_nm(lmax, el_dict, ref_elements, kernel_size, kernel_sparse_indices,
              power, power_ref, atom_counting, atomicindx, imol=0):

    k_NM = np.zeros(kernel_size, float)
    for iq, q in el_dict.items():
        for l in range(lmax[q]+1):
            msize = 2*l+1
            for iref in np.where(ref_elements==iq)[0]:
                block_ref = power_ref.block(spherical_harmonics_l=l, species_center=q)
                pos1 = block_ref.samples.position((iref,))
                vec_ref = block_ref.values[pos1]
                for iatq in range(atom_counting[iq]):
                    iat = atomicindx[iq,iatq]

                    block = power.block(spherical_harmonics_l=l, species_center=q)
                    pos2 = block.samples.position((imol,iat))
                    vec = block.values[pos2]

                    if l==0:
                        kern = np.dot(vec, vec_ref.T)**2
                    else:
                        ik0 = kernel_sparse_indices[iref,iatq,0]
                        kern = np.dot(vec, vec_ref.T) * k_NM[ik0]**0.5

                    ik = kernel_sparse_indices[iref,iatq,l]
                    k_NM[ik:ik+msize*msize] = kern.T.flatten()
    return k_NM


def kernel_mm(M, lmax, powerrefbase, ref_elements):

  llmax = max(lmax.values())
  k_MM = np.zeros((llmax+1, M*(2*llmax+1), M*(2*llmax+1)))
  power_ref = equistore.load(f'{powerrefbase}_{M}.npz')

  for q in set(ref_elements):
      for l in range(lmax[q]+1):
          ms = 2*l+1
          block = power_ref.block(spherical_harmonics_l=l, species_center=q)
          for iref1 in block.samples:
              pos1 = block.samples.position(iref1)
              vec1 = block.values[pos1]
              for iref2 in block.samples:
                  pos2 = block.samples.position(iref2)
                  vec2 = block.values[pos2]
                  k_MM[l, iref1[0]*ms:(iref1[0]+1)*ms, iref2[0]*ms:(iref2[0]+1)*ms] = vec1 @ vec2.T

  for l in range(llmax, -1, -1):
      ms = 2*l+1
      for iref1 in range(M):
          for iref2 in range(M):
              k_MM[l, iref1*ms:(iref1+1)*ms, iref2*ms:(iref2+1)*ms] *= k_MM[0, iref1, iref2]

  return k_MM


def kernel_for_mol(power_ref, power_file, kernel_file,
                   lmax, ref_elements, el_dict,
                   natoms, atom_counting, atomicindx):

    power = equistore.load(power_file)
    kernel_size, kernel_sparse_indices = kernel_nm_sparse_indices(natoms, lmax,
                                         ref_elements, el_dict, atom_counting)
    k_NM = kernel_nm(lmax, el_dict, ref_elements, kernel_size, kernel_sparse_indices,
                     power, power_ref, atom_counting, atomicindx)
    np.savetxt(kernel_file, k_NM)
