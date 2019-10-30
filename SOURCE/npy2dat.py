import sys
import time
import numpy as np

ndata = 1000 

for iconf in xrange(ndata):
    start = time.time()
    projs = np.load("PROJS_NPY/projections_conf"+str(iconf)+".npy")
    np.savetxt("PROJS_DAT/projections_conf"+str(iconf)+".dat", projs, fmt='%.06e')
    overlap = np.load("OVER_NPY/overlap_conf"+str(iconf)+".npy")
    np.savetxt("OVER_DAT/"+str(mol)+"/overlap_conf"+str(iconf)+".dat", np.concatenate(overlap), fmt='%.06e')
    print iconf,time.time()-start

