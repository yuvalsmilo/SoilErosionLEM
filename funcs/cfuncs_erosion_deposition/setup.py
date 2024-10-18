from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

##




##
ext_modules = [
    Extension(
        "cfuncs_ErosionDeposition",
        ["cfuncs_ErosionDeposition.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]



setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)

