from ase.data import atomic_numbers

def basis_read(filename):
  (basis, el_dict, Lmax, Nmax) = basis_read_full(filename)
  return (el_dict, Lmax, Nmax)

def basis_read_full(filename):

  with open(filename, "r") as f:
    lines = [ i.strip() for i in f.read().split('\n') ]

  el_dict = {}
  angular_momenta = {}
  basis = {}

  errormsg = 'you gave me a bad basis'

  nelements = 0
  q = None
  i = 0
  while i < len(lines):

    if len(lines[i]) > 2 and lines[i][0:2] == 'O-':
      q = lines[i].split(' ')[1]
      q = atomic_numbers[q.capitalize()]
      if q in angular_momenta:
        raise SystemExit(errormsg)
      el_dict[nelements] = q
      nelements += 1
      angular_momenta[q] = []
      basis          [q] = []
      i+=1
      continue

    elif len(lines[i]) == 0 or lines[i][0] == '#':
      i+=1
      continue

    elif lines[i].isdigit():
      if q == None or len(angular_momenta[q])>0:
        raise SystemExit(errormsg)
      nbf = int(lines[i])
      i+=1
      for j in range(nbf):
        [tmp, l, np] = [int(k) for k in filter(None, lines[i].split(' ') )];
        angular_momenta[q].append(l)
        i+=1

        gto = []
        for k in range(np):
          a, c = lines[i].split()
          gto.append([float(a), float(c)])
          i+=1
        basis[q].append([l, gto])

    else:
      raise SystemExit(errormsg)

  Lmax = {}
  Nmax = {}

  for q in angular_momenta:
    lmax = max(angular_momenta[q])
    Lmax[q] = lmax
    for l in range(0,lmax+1):
      Nmax[(q,l)] = angular_momenta[q].count(l)

  return (basis, el_dict, Lmax, Nmax)

def basis_print(basis):
  for q in basis:
    print(q)
    for gto in basis[q]:
      l = gto[0]
      prim = gto[1]
      n = len(prim)
      print ("l =", l, "n =", n)
      for i in prim:
        print(i)

