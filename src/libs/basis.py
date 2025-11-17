import numpy as np
from ase.data import atomic_numbers


def basis_read(filename):
    _, Lmax, Nmax = basis_read_full(filename)
    return Lmax, Nmax


def basis_read_full(filename):

    msg_incorrect = 'Incorrect basis file format!'
    msg_lorder    = 'Basis set should be in the ascending L order'

    with open(filename) as f:
        lines = [i.strip() for i in f.read().split('\n')]

    angular_momenta = {}
    basis = {}
    q = None
    i = 0
    while i<len(lines):

      if len(lines[i]) > 2 and lines[i][0:2] == 'O-':
          q = lines[i].split(' ')[1]
          q = atomic_numbers[q.capitalize()]
          if q in angular_momenta:
              raise SystemExit(msg_incorrect)
          angular_momenta[q] = []
          basis          [q] = []
          i += 1

      elif len(lines[i]) == 0 or lines[i][0] == '#':
          i += 1

      elif lines[i].isdigit():
          if q is None or len(angular_momenta[q])>0:
              raise SystemExit(msg_incorrect)
          nbf = int(lines[i])
          i += 1

          for _ in range(nbf):
              _, l, nprim = map(int, lines[i].split())
              angular_momenta[q].append(l)
              i += 1

              gto = []
              for _ in range(nprim):
                  gto.append(tuple(map(float, lines[i].split())))
                  i += 1
              basis[q].append((l, gto))

      else:
          raise SystemExit(msg_incorrect)

    Lmax = {}
    Nmax = {}
    for q, llist in angular_momenta.items():
        if llist == sorted(llist):
            SystemExit(msg_lorder)
        Lmax[q] = max(llist)
        for l, nsize in zip(*np.unique(np.array(angular_momenta[q]), return_counts=True), strict=True):
            Nmax[(q,l)] = nsize

    return basis, Lmax, Nmax


def basis_print(basis):
    for q in basis:
        print(q)
        for l, prim in basis[q]:
            n = len(prim)
            print(f'{l=} {n=}')
            for i in prim:
                print(i)
