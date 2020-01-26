import numpy as np
from power_spectra_lib import read_ps_1mol

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


def kernel_mm(M, llmax, powerrefbase):

  k_MM = np.zeros((llmax+1,M*(2*llmax+1),M*(2*llmax+1)),float)

  for l in range(llmax+1):
      power_ref = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");
      if l==0:
          for iref1 in range(M):
              for iref2 in range(M):
                  k_MM[l,iref1,iref2] = np.dot(power_ref[iref1],power_ref[iref2].T)**zeta
      else:
          nfeat = power_ref.shape[2]
          power_ref = power_ref.reshape(M*(2*l+1),nfeat)
          ms = 2*l+1
          for iref1 in range(M):
              for iref2 in range(M):
                  k_MM[l, iref1*ms:(iref1+1)*ms, iref2*ms:(iref2+1)*ms] = np.dot(power_ref[iref1*ms:(iref1+1)*ms], power_ref[iref2*ms:(iref2+1)*ms].T) *  k_MM[0,iref1,iref2]**((zeta-1.0)/zeta)
  return k_MM

