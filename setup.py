import os
import tarfile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Distutils import extension
import numpy as np
import urllib

src_directory = "robust_pn"

files = []
files.append("robust_pn.pyx")

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('robust_pn', sources=files, language='c++',
                    include_dirs=[ np.get_include(), src_directory],                    
                    library_dirs=[],
                    extra_compile_args=["-std=c++0x","-O2", "-msse2",'-fopenmp', "-fpermissive"],                    
                                              extra_link_args=['-fopenmp'])
        ]
    )
