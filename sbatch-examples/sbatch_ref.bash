#!/bin/bash -l

#SBATCH --job-name=ref
#SBATCH --ntasks=1
#SBATCH --mem=254000
#SBATCH --cpus-per-task=32
#SBATCH --exclude=node55,node06,node12,node05

module purge
#module load anaconda/2019.3/python-3.7
module load intelmpi/17.0.4

conda activate density
path=../GIT/SOURCE
time $path/environments.py
time $path/power_spectra_ref.py
