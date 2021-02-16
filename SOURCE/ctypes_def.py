import numpy

array_1d_int    = numpy.ctypeslib.ndpointer(dtype=numpy.uint32,  ndim=1, flags='CONTIGUOUS')
array_2d_int    = numpy.ctypeslib.ndpointer(dtype=numpy.uint32,  ndim=2, flags='CONTIGUOUS')
array_3d_int    = numpy.ctypeslib.ndpointer(dtype=numpy.uint32,  ndim=3, flags='CONTIGUOUS')
array_2d_double = numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2, flags='CONTIGUOUS')
array_3d_double = numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=3, flags='CONTIGUOUS')
array_4d_double = numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=4, flags='CONTIGUOUS')
array_5d_double = numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=5, flags='CONTIGUOUS')
