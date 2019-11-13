import numpy as np

def read_ps(psfilename, l, ndata, natmax, nspecies, atom_counting, atomicindx):

    power = np.load(psfilename)

    if l==0:
        nfeat = len(power[0,0])
        power_per_conf = np.zeros((ndata,natmax,nfeat),float)
    else:
        nfeat = len(power[0,0,0])
        power_per_conf = np.zeros((ndata,natmax,2*l+1,nfeat),float)

    # power spectrum
    for iconf in xrange(ndata):
        iat = 0
        for ispe in xrange(nspecies):
            for icount in xrange(atom_counting[iconf,ispe]):
                jat = atomicindx[iconf,ispe,icount]
                power_per_conf[iconf,jat] = power[iconf,iat]
                iat+=1
    return (nfeat, power_per_conf)
