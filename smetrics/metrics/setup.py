#!/usr/bin/env python

import os
import sys

base_path = os.path.abspath(os.path.dirname(__file__))

from restorationpy._build_utils import cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('metrics', parent_package, top_path)

    # submodules which have their own setup.py
    config.add_subpackage('tests')

    #config.add_extension('_coefficients',
    #                     sources=['_coefficients.pyx'],
    #                     include_dirs=[numpy.get_include()],
    #                     )

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
