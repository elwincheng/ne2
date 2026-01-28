# setup.py
import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Platform-specific compile flags
if platform.system() == "Darwin":
    # macOS: clang doesn't support -fopenmp directly without libomp
    # Build without OpenMP for simplicity (still fast due to C++)
    extra_compile = ["-O3", "-std=c++17"]
    extra_link = []
elif platform.system() == "Windows":
    extra_compile = ["/O2", "/std:c++17", "/openmp"]
    extra_link = []
else:
    # Linux: GCC supports OpenMP
    extra_compile = ["-O3", "-std=c++17", "-fopenmp"]
    extra_link = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "remove_d",
        ["remove_d.cpp"],
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
    ),
]

setup(
    name="example_module",
    version="1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A pybind11 example project",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

# python setup.py build_ext --inplace