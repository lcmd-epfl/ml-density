mols="quantum-chemistry/geometries_dimers/*.xyz quantum-chemistry/geometries_monomers/*.xyz"

cat ${mols} > ALL.xyz

mkdir ALL_J ALL_Cj ALL_S ALL_Cs ALL_Csn ALL_Cjn
k=0
for mol in $mols; do
  name=${mol##*/}
  name=${name%%.xyz*}
  printf "%5d   %s\n" $k $name

  j=quantum-chemistry/calculations_pyscf/${name}/Jmat.npy
  cj=quantum-chemistry/calculations_pyscf/${name}/coeffs.dat
  cjn=quantum-chemistry/s_j_fitting_correct_N/${name}.cjn.dat
  s=quantum-chemistry/calculations_q/${name}.s.npy
  cs=quantum-chemistry/calculations_q/${name}.cs.dat
  csn=quantum-chemistry/s_j_fitting_correct_N/${name}.csn.dat
  ln -s $(readlink -m ${j}) "ALL_J/mol_${k}.npy"
  ln -s $(readlink -m ${s}) "ALL_S/mol_${k}.npy"
  ln -s $(readlink -m ${cj}) "ALL_Cj/mol_${k}.dat"
  ln -s $(readlink -m ${cs}) "ALL_Cs/mol_${k}.dat"
  ln -s $(readlink -m ${cjn}) "ALL_Cjn/mol_${k}.dat"
  ln -s $(readlink -m ${csn}) "ALL_Csn/mol_${k}.dat"
  k=$(($k+1))
done

