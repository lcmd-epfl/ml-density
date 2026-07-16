path=../src

# Requires full_gpr = true in config.txt

$path/preprocess.py
$path/training_selection.py
$path/power_spectra.py
$path/select_reference_environments.py
$path/power_spectra_reference.py
$path/kernel_mm.py
$path/kernel_nm.py
$path/get_matrices.py -b   # also computes the A-vector under full_gpr, no separate call needed
$path/regression.py
$path/prediction.py
$path/variance.py
$path/compute_error.py > error.txt

# $path/power_spectra.py --extra
# $path/kernel_nm.py --extra
# $path/prediction.py --extra

# Visualize a single molecule (density or variance field):
# $path/extract_cube.py --mol 0
# $path/extract_cube.py --std --mol 0
# $path/extract_cube.py --diff --mol 0
