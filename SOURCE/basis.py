from ase.data import atomic_numbers

def basis_read(filename):
  (basis, spe_dict, Lmax, Nmax) = basis_read_full(filename)
  return (spe_dict, Lmax, Nmax)

def basis_read_full(filename):

  f = open(filename, "r")
  lines = [ i.strip() for i in f.read().split('\n') ];
  f.close()

  spe_dict = {}
  angular_momenta = {}
  basis = {}

  errormsg = 'you gave me a bad basis'

  element = None
  nelements = 0
  i = 0
  while i < len(lines):

    if len(lines[i]) > 2 and lines[i][0:2] == 'O-':
      element = lines[i].split(' ')[1]
      element = atomic_numbers[element]
      if element in angular_momenta.keys():
        raise SystemExit(errormsg)
      spe_dict[nelements] = element
      nelements+=1
      angular_momenta[ element ] = []
      basis          [ element ] = []
      i+=1
      continue

    elif len(lines[i]) == 0 or lines[i][0] == '#':
      i+=1
      continue

    elif lines[i].isdigit():
      if element == None or len(angular_momenta[element])>0:
        raise SystemExit(errormsg)
      nbf = int(lines[i])
      i+=1
      for j in range(nbf):
        [tmp, l, np] = [int(k) for k in filter(None, lines[i].split(' ') )];
        angular_momenta[element].append(l)
        i+=1

        gto = []
        for ii in range(np):
          a, c = lines[i].split()
          gto.append([float(a), float(c)])
          i+=1
        basis[element].append([l, gto])

    else:
      raise SystemExit(errormsg)

  Lmax = {}
  Nmax = {}

  for q in angular_momenta.keys():
    lmax = max(angular_momenta[q])
    Lmax[q] = lmax
    for l in range(0,lmax+1):
      Nmax[(q,l)] = angular_momenta[q].count(l)

  return (basis, spe_dict, Lmax, Nmax)

def basis_print(basis):
  for key in basis.keys():
    print(key)
    for gto in basis[key]:
      l = gto[0]
      prim = gto[1]
      n = len(prim)
      print ("l =", l, "n =", n)
      for i in prim:
        print(i)

