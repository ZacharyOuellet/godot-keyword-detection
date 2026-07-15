from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "keyword_detection",
        sources=[
            "bindings.cpp",
            "../src/mfcc_processor_core.cpp",
            "../src/pncc_processor_core.cpp",
            "../src/dtw_matcher_core.cpp",
        ],
        include_dirs=[".", "../include"],  # so `#include "keyword_detection/..."` resolves
        cxx_std=17,
    ),
]

setup(
    name="keyword_detection",
    version="0.1.0",
    description="MFCC/PNCC feature extraction + DTW template matching "
                 "(pybind11 bindings around the same core used by the Godot extension)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
)
