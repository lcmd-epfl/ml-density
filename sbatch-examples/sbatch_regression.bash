#!/bin/bash -l

#SBATCH --job-name=inversion
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=26

module purge
#module load anaconda/2019.3/python-3.7
#module load intelmpi/17.0.4

conda activate density
path=../GIT/SOURCE

for config in config_{elec,spin}_{ps-nonnormalized,norm-zeroed}.txt ; do
    time python $path/regression.py --config=$config
    time python $path/prediction.py --config=$config
    time python $path/compute_error.py --config=$config > ${config}_error
done
