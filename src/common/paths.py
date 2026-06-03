"""
Centralized project paths and environment configuration.

Usage:
    from src.common.paths import get_project_root, get_data_dir, get_output_dir

All hardcoded /home/bendm/... paths are replaced with environment-variable-backed
defaults so the codebase is portable across machines.

Set these environment variables to override defaults:
    THESIS_PROJECT_ROOT   — root of the code/ repository
    THESIS_DATA_DIR       — root of datasets/ directory
    THESIS_OUTPUT_DIR     — root for outputs/results
    THESIS_CHECKPOINTS_DIR — root for model checkpoints
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory (code/)."""
    env = os.environ.get("THESIS_PROJECT_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the data directory (datasets/), defaulting to ../datasets from project root."""
    env = os.environ.get("THESIS_DATA_DIR")
    if env:
        return Path(env)
    return get_project_root().parent / "datasets"


def get_output_dir() -> Path:
    """Return the output directory, defaulting to <project_root>/outputs."""
    env = os.environ.get("THESIS_OUTPUT_DIR")
    if env:
        return Path(env)
    return get_project_root() / "outputs"


def get_checkpoints_dir() -> Path:
    """Return the checkpoints directory, defaulting to <project_root>/checkpoints."""
    env = os.environ.get("THESIS_CHECKPOINTS_DIR")
    if env:
        return Path(env)
    return get_project_root() / "checkpoints"


def setup_python_path():
    """Add project root and src/ to sys.path so imports work from any location.

    Call this once at the top of any entry-point script:
        from src.common.paths import setup_python_path
        setup_python_path()
    """
    import sys

    root = str(get_project_root())
    src = str(get_project_root() / "src")

    for p in (root, src):
        if p not in sys.path:
            sys.path.insert(0, p)
