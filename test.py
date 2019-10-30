#!/usr/bin/python

from basis import basis_read

(spe_dict, lmax, nmax) = basis_read("cc-pvqz-jkfit.1.d2k")

print spe_dict
print lmax
print nmax

print max(lmax.values())
print max(nmax.values())
