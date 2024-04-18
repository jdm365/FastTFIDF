from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import os

MODULE_NAME = "fast_tfidf"

## Optionally choose compiler ##

COMPILER = "clang++"
## COMPILER = "g++"
os.environ["CXX"] = COMPILER

COMPILER = os.environ["CXX"]

COMPILER_FLAGS = [
    "-std=c++17",
    "-O3",
    "-Wall",
    "-Wextra",
    "-march=native",
    "-ffast-math",
]

OS = os.uname().sysname

LINK_ARGS = [
    "-lc++",
    "-lc++abi",
    "-L/usr/local/lib",
]

if COMPILER == "clang++":
    COMPILER_FLAGS += ["-stdlib=libc++"]


extensions = [
    Extension(
        MODULE_NAME,
        sources=["fast_tfidf/fast_tfidf.pyx", "fast_tfidf/engine.cpp"],
        extra_compile_args=COMPILER_FLAGS,
        language="c++",
        include_dirs=[np.get_include(), "fast_tfidf"],
        extra_link_args=LINK_ARGS,
    ),
]

setup(
    name=MODULE_NAME,
    ext_modules=cythonize(extensions),
)
