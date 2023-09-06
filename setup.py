from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys
import os

dirname =  os.path.abspath(os.path.dirname(__file__))
print("xxxxx xxxx xxxx xxx ")
print("dirname", dirname)
print(os.getcwd())
print(os.listdir())
print(os.listdir('./src'))
print("xxxxx xxxx xxxx xxx ")
__version__ = "0.0.1"


ext_modules = [
    Pybind11Extension("MonotoneScheme",
        ["src/main.cpp"],
        define_macros      = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="MonotoneScheme",
    version=__version__,
    author="Wonjun Lee",
    author_email="wlee@ucla.edu",
    description="Python wrapper for the monotone discretization problem in 2D and 3D Cartesian grids (join work with Jeff Calder (UMN))",
    long_description="""
        (the details will be added)
        """,
    long_description_content_type="text/markdown",
    url="https://github.com/wonjunee/monotone-scheme",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"],
    install_requires=[  'numpy', 
                        'scipy', 
                        'scikit-learn', 
                        'matplotlib'],
    python_requires='>=3.6',
    zip_safe=False,
)
