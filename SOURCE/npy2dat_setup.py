from distutils.core import setup, Extension

npy2dat = Extension('npy2dat',
                    sources=['npy2dat.c'],
                    extra_compile_args = ["--std=gnu11"])

setup(ext_modules=[npy2dat])
