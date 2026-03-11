# -*- coding: utf-8 -*-
from importlib.metadata import version, PackageNotFoundError

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from ppscore.calculation import score, predictors, matrix
