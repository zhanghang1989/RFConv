import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))

cpu = load('enclib_cpu', [
        os.path.join(cwd, 'operator_cpu.cpp'),
        os.path.join(cwd, 'rectify_cpu.cpp'),
    ], build_directory=cwd, verbose=False)

if torch.cuda.is_available():
    gpu = load('enclib_gpu', [
            os.path.join(cwd, 'operator_cuda.cpp'),
            os.path.join(cwd, 'rectify_cuda.cu'),
        ], extra_cuda_cflags=["--expt-extended-lambda"],
        build_directory=cwd, verbose=False)
