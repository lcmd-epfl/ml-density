import numpy

array_1d_int = numpy.ctypeslib.ndpointer(dtype=numpy.uint32, ndim=1, flags='CONTIGUOUS')
array_2d_int = numpy.ctypeslib.ndpointer(dtype=numpy.uint32, ndim=2, flags='CONTIGUOUS')
array_3d_int = numpy.ctypeslib.ndpointer(dtype=numpy.uint32, ndim=3, flags='CONTIGUOUS')
