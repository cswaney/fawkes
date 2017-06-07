from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize('**/*.pyx')

for e in ext_modules:
    e.extra_compile_args.extend(['-fopenmp'])
    e.extra_link_args.extend(['-fopenmp'])

setup(name='fawkes',
      author='Colin Swaney',
      author_email='colinswaney@gmail.com',
      version='0.1',
      ext_modules=ext_modules,
      install_requires=['numpy', 'cython'],
      include_dirs=[np.get_include(),],
      test_suite='nose.collector',
      tests_require=['nose'],
      packages=['fawkes']
)
