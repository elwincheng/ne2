# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "remove_d",
        ["remove_d.cpp"],
        # Example of adding a custom compiler argument:
        # if platform.system() == "Windows":
        #    extra_compile_args=["/std:c++14"]
        # else:
        #    extra_compile_args=["-std=c++14"],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
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