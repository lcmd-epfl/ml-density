module purge

path=src

# 2) Define sparse set {M} from scalar SOAP-vector
#    Extract PS for {M}

time python3 $path/power_spectra_ref.py
echo

# 3) Build K_NM

time python3 $path/kernels.py
echo

# 4) Build K_MM

time python3 $path/rmatrix.py
echo

# 5) Baseline coefficients and compute projections

time python3 $path/project.py
echo

# 6) Make a training set

python3 $path/training_selection.py
echo

# 7) Compute A and B

time mpirun -np 9 python3 $path/get_matrices.py
time mpirun -np 9 python3 $path/get_matrices.py b
echo

# 8) REGRESSION!

time python $path/regression.py
echo

# 9) predictions

time python $path/prediction.py
echo

# 10) compute error

time python $path/compute_error.py

