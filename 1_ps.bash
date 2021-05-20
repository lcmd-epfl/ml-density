module purge
module load anaconda/5.2.0/python-2.7
source /home/afabrizi/MyQuantumChemSoftwares/SOAPfast/SOAPFAST-master/env.sh

# 1) Generate power spectra

XYZPATH=$(ls quantum-chemistry/geometries/*.xyz | sort -V)
OUTPUT=PS/PS
ALLMOL=ALL.xyz
SPECIES="H C N O"

LOUTMAX=5
RCUT=4.0
SIGMA=0.3
LCUT=6
NCUT=8

for L in `seq 0 $LOUTMAX`; do
  K=0
  for MOL in $XYZPATH; do
    if [ ! -f ${OUTPUT}${L}_${K}.npy ] ; then
      echo $L $K
      sbatch --job-name=${L}_${K} --mem=2000 -e /dev/null -o /dev/null --wrap "sagpr_get_PS -n $NCUT -l $LCUT -rc $RCUT -sg $SIGMA -lm $L -f $MOL -o ${OUTPUT}${L}_${K} -s $SPECIES -c $SPECIES"
      sleep 0.1
    fi
    K=$(($K+1))
  done
done

sbatch --mem=66000 --cpus-per-task=16 --wrap "sagpr_get_PS -n $NCUT -l $LCUT -rc $RCUT -sg $SIGMA -lm 0 -f $ALLMOL -o ${OUTPUT}0 -s $SPECIES -c $SPECIES"
