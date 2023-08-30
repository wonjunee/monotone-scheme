from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"


ext_modules = [
    Pybind11Extension("monotone",
        ["src/main.cpp"],
        define_macros      = [('VERSION_INFO', __version__)],
        extra_compile_args = ['-std=c++14'],
        ),
]

setup(
    name="monotone",
    version=__version__,
    author="Wonjun Lee",
    author_email="wlee@ucla.edu",
    description="Python wrapper for the monotone discretization problem in 3D Cartesian grids (join work with Jeff Calder (UMN))",
    long_description="""
        (the details will be added)
        """,
    long_description_content_type="text/markdown",
    url="https://github.com/wonjunee/monotone-scheme",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)