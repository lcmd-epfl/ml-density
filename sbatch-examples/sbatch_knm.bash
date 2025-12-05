#!/bin/bash -l

#SBATCH --job-name=knm
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node55,node06,node12,node05

module purge
#module load anaconda/2019.3/python-3.7
module load intelmpi/17.0.4

conda activate density
path=../GIT/SOURCE
time srun $path/kernels.py
