import numpy as np
import equistore

# SOAP parameters
zeta = 2.0


def kernel_nm_sparse_indices(M, natoms, llmax, lmax,
    ref_elements, el_dict, atom_counting):

    kernel_size = 0
    kernel_sparse_indices = np.zeros((M,natoms,llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in range(M):
        iel = ref_elements[iref]
        q   = el_dict[iel]
        for l in range(lmax[q]+1):
            msize = 2*l+1
            for im in range(msize):
                for iat in range(atom_counting[iel]):
                    for imm in range(msize):
                        kernel_sparse_indices[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1
    return kernel_size, kernel_sparse_indices


def kernel_nm(M, llmax, lmax, nel,
    el_dict, ref_elements,
    kernel_size, kernel_sparse_indices,
    power, power_ref,
    atom_counting, atomicindx,
    imol=None):

    k_NM = np.zeros(kernel_size,float)
    for iref in range(M):
        iel = ref_elements[iref]
        q   = el_dict[iel]
        for iatq in range(atom_counting[iel]):
            iat = atomicindx[iel,iatq]
            ik0 = kernel_sparse_indices[iref,iatq,0,0,0]
            for l in range(lmax[q]+1):

                if imol is None:
                    powert = power[l][iat]
                else:
                    powert = power[l][imol,iat]
                powerr = power_ref[l][iref]

                msize = 2*l+1
                if l==0:
                    ik = kernel_sparse_indices[iref,iatq,l,0,0]
                    k_NM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in range(msize):
                        for im2 in range(msize):
                            ik = kernel_sparse_indices[iref,iatq,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    return k_NM


def kernel_nm_new(lmax, nel,
    el_dict, ref_elements,
    kernel_size, kernel_sparse_indices,
    power, power_ref,
    atom_counting, atomicindx,
    imol=0):

    k_NM = np.zeros(kernel_size, float)
    for iq, q in el_dict.items():
        for l in range(lmax[q]+1):
            for iref in np.where(ref_elements==iq)[0]:
                block_ref = power_ref.block(spherical_harmonics_l=l, species_center=q)
                pos1 = block_ref.samples.position((iref,))
                vec_ref = block_ref.values[pos1]
                for iatq in range(atom_counting[iq]):
                    iat = atomicindx[iq,iatq]
                    ik0 = kernel_sparse_indices[iref,iatq,0,0,0]

                    block = power.block(spherical_harmonics_l=l, species_center=q)
                    pos2 = block.samples.position((imol,iat))
                    vec = block.values[pos2]

                    if l==0:
                        kern = np.dot(vec, vec_ref.T)**2
                    else:
                        kern = np.dot(vec, vec_ref.T) * k_NM[ik0]**0.5

                    msize = 2*l+1
                    for im1 in range(msize):
                        for im2 in range(msize):
                            ik = kernel_sparse_indices[iref,iatq,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    return k_NM



def kernel_mm_new(M, lmax, powerrefbase, ref_elements):

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
