#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("speedup", ["speedup.pyx"],
          runtime_library_dirs=['/usr/local/lib'],
          extra_compile_args=[ '-I', 
'/Library/Python/2.6/site-packages/numpy-1.4.0-py2.6-macosx-10.6-universal.egg/numpy/core/include/']),

              Extension("speedup2", ["speedup2.c"])]


setup(
  name = 'speedup',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
