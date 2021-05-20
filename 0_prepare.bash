mols=$(ls quantum-chemistry/geometries/*.xyz | sort -V)

cat ${mols} > ALL.xyz

mkdir ALL_C ALL_S
k=0
for mol in $mols; do
  printf "%5d   %s\n" $k $mol
  c=${mol}.coef.dat
  s=${mol}.smat.npy
  ln -s $(readlink -m ${c}) "ALL_C/mol_${k}.dat"
  ln -s $(readlink -m ${s}) "ALL_S/mol_${k}.npy"
  k=$(($k+1))
done

