from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.7"


ext_modules = [
    Pybind11Extension("MonotoneScheme",
        ["src/main.cpp"],
        define_macros      = [('VERSION_INFO', __version__)],
        extra_compile_args = ['-std=c++14'],
        include_dirs=["src/"],  # -I
        ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MonotoneScheme",
    version=__version__,
    author="Wonjun Lee",
    author_email="wlee@ucla.edu",
    description="Python wrapper for C++ codes for the monotone scheme for curvature motion PDEs",
    long_description=long_description,
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
