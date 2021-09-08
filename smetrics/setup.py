"""Metrics implementation.

Implementation of the image quality metrics. Each metric should respect the
following API

   metric = MyMetric(param1, param2, ...)
   value = metric.run()
"""

import sys
import os
from numpy.distutils.misc_util import Configuration
from smetrics._build_utils import cythonize_extensions


def configuration(parent_package='', top_path=None):
    """Configure the package

    Parameters
    ----------
    parent_package: str
        name of the parent package
    top_path: str
        Top path for cython

    """

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('smetrics', parent_package, top_path)

    # submodules with build utilities
    config.add_subpackage('__check_build')
    config.add_subpackage('_build_utils')

    # submodules which have their own setup.py
    config.add_subpackage('metrics')

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
