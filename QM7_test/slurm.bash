#!/bin/bash -l
#
# Usage: bash slurm.bash [basis_name] (dz (default), tz or aqz)

basis=${1:-dz}
sbatch --job-name $basis --output logs/$basis-%j.out run.bash config_$basis.txt
