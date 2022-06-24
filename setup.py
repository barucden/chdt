import os

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension

mod = CppExtension('chdt',
        sources=[
            'src/chdt.cpp',
            'src/chdt_cpu.cpp'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-lgomp'])

setup(
    name='chdt',
    ext_modules=[mod],
    cmdclass={
        'build_ext': BuildExtension
    })

