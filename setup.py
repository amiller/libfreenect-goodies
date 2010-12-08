#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("speedup_ctypes", ["speedup_ctypes.c"], extra_compile_args=['-g','-O3'])]


setup(
  name = 'speedup',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
