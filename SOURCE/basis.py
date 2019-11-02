def basis_read(filename):

  f = open(filename, "r")
  lines = [ i.strip() for i in f.read().split('\n') ];
  f.close()

  spe_dict = {}
  angular_momenta = {}

  errormsg = 'you gave me a bad basis'

  element = None
  nelements = 0
  i = 0
  while i < len(lines):

    if len(lines[i]) > 2 and lines[i][0:2] == 'O-':
      element = lines[i].split(' ')[1]
      if element in angular_momenta.keys():
        raise SystemExit(errormsg)
      spe_dict[nelements] = element
      nelements+=1
      angular_momenta[ element ] = []
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
        i += np+1

    else:
      raise SystemExit(errormsg)

  Lmax = {}
  Nmax = {}

  for q in angular_momenta.keys():
    lmax = max(angular_momenta[q])
    Lmax[q] = lmax
    for l in range(0,lmax+1):
      Nmax[(q,l)] = angular_momenta[q].count(l)

  return (spe_dict, Lmax, Nmax)

