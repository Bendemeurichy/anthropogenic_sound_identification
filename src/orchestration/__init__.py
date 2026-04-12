"""
Sound Separation Pipeline Orchestration.

This module provides a simplified interface for managing sound separation projects,
designed for biologists and end-users who need to train and run inference on
custom sound classes without dealing with ML complexity.

Quick Start:
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
    >>> results["bird"].save("bird_only.wav")

See Also:
    - SoundClass: Define a sound class to train on
    - Project: Manage a sound separation project
    - TrainingPreset: Pre-configured training settings
"""

from .sound_class import SoundClass
from .project import Project, ProjectSettings
from .presets import TrainingPreset, get_preset

__all__ = [
    "SoundClass",
    "Project", 
    "ProjectSettings",
    "TrainingPreset",
    "get_preset",
]
