"""
Sound Separation Pipeline Orchestration.

This module provides a simplified interface for managing sound separation projects,
designed for biologists and end-users who need to train and run inference on
custom sound classes without dealing with ML complexity.

Quick Start (Python):
    >>> from orchestration import Project, SoundClass
    >>>
    >>> # Create a project
    >>> project = Project.create("my_bird_study", base_dir="./projects")
    >>>
    >>> # Add sound classes to detect
    >>> project.add_class(SoundClass(
    ...     name="bird",
    ...     labels=["bird", "birdsong", "bird_call"],
    ...     description="Bird vocalizations"
    ... ))
    >>> project.add_class(SoundClass(
    ...     name="airplane",
    ...     labels=["airplane", "aircraft", "plane"],
    ...     description="Aircraft noise"
    ... ))
    >>>
    >>> # Train the model
    >>> project.train(epochs=100)
    >>>
    >>> # Run inference on audio files
    >>> results = project.separate("field_recording.wav")
    >>> results["bird"]  # Bird-only audio tensor

Quick Start (CLI):
    # Create a project
    python -m orchestration create my_bird_study ./projects

    # Add classes
    python -m orchestration add-class my_bird_study bird "bird,birdsong,bird_call"
    python -m orchestration add-class my_bird_study airplane "airplane,aircraft,plane"

    # Train
    python -m orchestration train my_bird_study --data-path metadata.csv

    # Separate
    python -m orchestration separate my_bird_study recording.wav --output out/

See Also:
    - SoundClass: Define a sound class to train on
    - Project: Manage a sound separation project
    - TrainingPreset: Pre-configured training settings
"""

from .sound_class import SoundClass
from .project import Project, ProjectSettings
from .presets import TrainingPreset, HardwareSettings, get_preset, list_presets

__all__ = [
    "SoundClass",
    "Project",
    "ProjectSettings",
    "TrainingPreset",
    "HardwareSettings",
    "get_preset",
    "list_presets",
]
