"""SuDoRM-RF model package for class-of-interest audio separation."""

import sys
from pathlib import Path

# Ensure the sudormrf package directory is on sys.path so that submodules
# can resolve imports like ``from base.sudo_rm_rf...`` regardless of how
# the package is loaded (direct script execution or package import).
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
