import os

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
import distutils.util
import numpy
from numpy.distutils.system_info import blas_info

# includes numpy : package numpy.distutils , numpy.get_include()
# python setup.py build
# python setup.py install --prefix=dist,
incs = ['.'] + [numpy.get_include(), get_python_inc()] + blas_info().get_include_dirs()

osname = distutils.util.get_platform()
# cc_flags = ['-fPIC', '-fopenmp', '-Wunused-variable', '-m64']
cc_flags = ['-fPIC', '-Wall', '-fopenmp', '-std=c++11', '-lm', '-Wfatal-errors']
for _ in numpy.__config__.blas_opt_info.get("extra_compile_args", []):
    if _ not in cc_flags:
        cc_flags.append(_)
for _ in numpy.__config__.lapack_opt_info.get("extra_compile_args", []):
    if _ not in cc_flags:
        cc_flags.append(_)

link_flags = ['-fopenmp']
for _ in numpy.__config__.blas_opt_info.get("extra_link_args", []):
    if _ not in link_flags:
        link_flags.append(_)
for _ in numpy.__config__.lapack_opt_info.get("extra_link_args", []):
    if _ not in link_flags:
        link_flags.append(_)

libs = ['stdc++', 'mkl_rt', 'iomp5']
libdirs = numpy.distutils.system_info.blas_info().get_lib_dirs()

miso = Extension(
    'miso_svm._miso',
    sources = ['miso.cpp'],
    include_dirs = incs,
    extra_compile_args = ['-DINT_64BITS', '-DAXPBY', '-DHAVE_MKL'] + cc_flags,
    library_dirs = libdirs,
    libraries = libs,
    extra_link_args = link_flags,
    language = 'c++',
    depends = ['cblas_alt_template.h', 'cblas_defvar.h',
                'common.h', 'ctypes_utils.h', 'linalg.h',
                'list.h', 'misc.h', 'svm.h', 'utils.h'],
)


setup ( name = 'miso_svm',
        version= '1.0',
        description='Python interface for MISO SVM classifier',
        author = 'Ghislain Durif',
        author_email = 'ckn.dev@inria.fr',
        url = None,
        license='GPLv3',
        ext_modules = [miso,],
        packages = ['miso_svm'],

        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 4 - Beta',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],

        # keywords='optimization',
        # install_requires=['numpy', 'scipy', 'scikit-learn'],
)
