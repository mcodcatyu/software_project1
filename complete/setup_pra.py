from setuptools import setup
from Cython.Build import cythonize
import numpy 

setup(
    ext_modules = cythonize("LLcython_pra.pyx", annotate=True),
    include_dirs = [numpy.get_include()]
     
)