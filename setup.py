from __future__ import annotations

import os
import os.path as osp
import glob
import platform
import sys
from itertools import product

from setuptools import find_packages, setup, Extension
import torch
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

__version__ = "0.1.0"
URL = "https://github.com/ken2403/pytorch_graph_utils"

# check device
WITH_CUDA = False
if torch.cuda.is_available():
    WITH_CUDA = CUDA_HOME is not None or torch.version.hip
devices = ["cpu", "cuda"] if WITH_CUDA else ["cpu"]
if os.getenv("FORCE_CUDA", "0") == "1":
    devices = ["cpu", "cuda"]
if os.getenv("FORCE_ONLY_CUDA", "0") == "1":
    devices = ["cuda"]
if os.getenv("FORCE_ONLY_CPU", "0") == "1":
    devices = ["cpu"]


def get_extensions() -> list[Extension]:
    extensions = []

    extensions_dir = osp.join("csrc")
    main_files = glob.glob(osp.join(extensions_dir, "*.cpp"))
    # remove generated 'hip' files, in case of rebuilds
    main_files = [path for path in main_files if "hip" not in path]

    for main, device in product(main_files, devices):
        define_macros = [("WITH_PYTHON", None)]
        undef_macros = []

        if sys.platform == "win32":
            define_macros += [("torchgraphutils_EXPORTS", None)]

        extra_compile_args = {"cxx": ["-O3"]}
        if not os.name == "nt":  # Not on Windows:
            extra_compile_args["cxx"] += ["-Wno-sign-compare"]
        extra_link_args = ["-s"]

        info = parallel_info()
        if (
            "backend: OpenMP" in info
            and "OpenMP not found" not in info
            and sys.platform != "darwin"
        ):
            extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
            if sys.platform == "win32":
                extra_compile_args["cxx"] += ["/openmp"]
            else:
                extra_compile_args["cxx"] += ["-fopenmp"]
        else:
            print("Compiling without OpenMP...")

        # Compile for mac arm64
        if sys.platform == "darwin" and platform.machine() == "arm64":
            extra_compile_args["cxx"] += ["-arch", "arm64"]
            extra_link_args += ["-arch", "arm64"]

        if device == "cuda":
            define_macros += [("WITH_CUDA", None)]
            nvcc_flags = os.getenv("NVCC_FLAGS", "")
            nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
            nvcc_flags += ["-O3"]
            if torch.version.hip:
                # USE_ROCM was added to later versions of PyTorch.
                # Define here to support older PyTorch versions as well:
                define_macros += [("USE_ROCM", None)]
                undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
            else:
                nvcc_flags += ["--expt-relaxed-constexpr"]
            extra_compile_args["nvcc"] = nvcc_flags

        name = main.split(os.sep)[-1][:-4]
        sources = [main]

        paths = glob.glob(osp.join(extensions_dir, "cpu", name, "*.cpp"))
        sources += [path for path in paths if "hip" not in path]

        paths = glob.glob(osp.join(extensions_dir, "cuda", name, "*.cu"))
        if device == "cuda":
            sources += [path for path in paths]

        Extension = CppExtension if device == "cpu" else CUDAExtension
        extension = Extension(
            f"torch_graph_utils._{name}_{device}",
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        extensions += [extension]

    return extensions


install_requires = [
    "torch>=1.7",
]

test_requires = [
    "pytest",
    "pytest-cov",
]

dev_requires = test_requires + [
    "pre-commit",
]

# work-around hipify abs paths
include_package_data = True
if torch.cuda.is_available() and torch.version.hip:
    include_package_data = False

setup(
    name="torch_graph_utils",
    version=__version__,
    description="PyTorch Extension for graph utilities",
    author="Kento Nishio",
    author_email="knishio@iis.u-tokyo.ac.jp",
    url=URL,
    keywords=["pytorch", "graph", "distance"],
    python_requires=">=3.7",
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True, use_ninja=False
        )
    },
    extras_require={
        "test": test_requires,
        "dev": dev_requires,
    },
    packages=find_packages(),
    include_package_data=include_package_data,
)
