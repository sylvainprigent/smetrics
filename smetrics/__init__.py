"""
Bio-image analysis module for Python
====================================

smetrics is a Python module integrating classical image analysis
algorithms based on the scientific ecosystem in python  (numpy, scipy,
matplotlib, scikit-image, scikit-learn).

See github.com/simglib for complete documentation.
"""
import sys
import logging
import os
import random
import numpy as np


from ._config import get_config, set_config, config_context

logger = logging.getLogger(__name__)


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.1.0'


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subcription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of smetrics when
    # the binaries are not built
    # mypy error: Cannot determine type of '__smetrics_SETUP__'
    __smetrics_SETUP__  # type: ignore
except NameError:
    __smetrics_SETUP__ = False

if __smetrics_SETUP__:
    sys.stderr.write('Partial import of smetrics during the build '
                     'process.\n')
    # We are not importing the rest of smetrics during the build
    # process, as it may not be compiled yet
else:
    # `_distributor_init` allows distributors to run custom init code.
    # For instance, for the Windows wheel, this is used to pre-load the
    # vcomp shared library runtime for OpenMP embedded in the simglib/.libs
    # sub-folder.
    # It is necessary to do this prior to importing show_versions as the
    # later is linked to the OpenMP runtime to make it possible to introspect
    # it and importing it first would fail if the OpenMP dll cannot be found.
    from . import _distributor_init  # noqa: F401
    from . import __check_build  # noqa: F401
    #from .base import clone
    #from .utils._show_versions import show_versions

    __all__ = ['metrics']


def setup_module():
    """Fixture for the tests to assure globally controllable seeding of RNGs"""

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get('smetrics_SEED', None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
