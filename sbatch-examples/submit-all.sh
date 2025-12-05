j0=$(sbatch sbatch_soap.bash | cut -f4 -d' ')

j1=$(sbatch --dependency=afterany:$j0 sbatch_ref.bash | cut -f4 -d' ')

j2=$(sbatch --dependency=afterany:$j1 --job-name=rmatrix --mem=254000 --cpus-per-task=24 --wrap 'conda activate qstack; time ../GIT/SOURCE/rmatrix.py' | cut -f4 -d' ')
j3=$(sbatch --dependency=afterany:$j1 sbatch_knm.bash | cut -f4 -d' ')

j4=$(sbatch --dependency=afterany:$j3 --job-name=avec --cpus-per-task=1 --mem=10GB --wrap 'conda activate qstack; time ../GIT/SOURCE/get_matrices.py' | cut -f4 -d' ')
j5=$(sbatch --dependency=afterany:$j3 sbatch_gm.bash | cut -f4 -d' ')

j6=$(sbatch --dependency=afterany:$j2:$j4:$j5 sbatch_regression.bash | cut -f4 -d' ')
