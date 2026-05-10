"""
Sound class definition for the orchestration layer.

A SoundClass represents a category of sounds that the model should learn to
separate from the audio mixture (e.g., "bird", "airplane", "rain").
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class SoundClass:
    """Defines a sound class (Class of Interest) for separation.
    
    A SoundClass represents one type of sound the model should learn to isolate.
    Multiple label strings can map to the same class (e.g., "bird", "birdsong",
    "bird_call" all map to the "bird" class).
    
    Attributes:
        name: Short identifier for this sound class. Used as the prompt name
              and in output file naming. Should be lowercase, no spaces.
              Examples: "bird", "airplane", "rain"
        labels: List of label strings from your metadata CSV that belong to
                this class. Case-sensitive matching against the 'label' column.
                Examples: ["bird", "Bird", "birdsong", "bird_call"]
        description: Human-readable description of what this class contains.
                    Useful for documentation and UI display.
        dataset_filter: Optional filter to only include labels from a specific
                       dataset (substring match on 'dataset' column). Leave
                       empty to include matching labels from all datasets.
        color: Optional hex color for visualization (e.g., "#FF5733").
               Will be auto-assigned if not specified.
    
    Example:
        >>> bird_class = SoundClass(
        ...     name="bird",
        ...     labels=["bird", "Bird", "birdsong", "bird_call", "bird_chirp"],
        ...     description="Bird vocalizations including songs, calls, and chirps",
        ... )
        >>> 
        >>> airplane_class = SoundClass(
        ...     name="airplane",
        ...     labels=["airplane", "aircraft", "jet", "propeller"],
        ...     description="Aircraft engine noise",
        ...     dataset_filter="aerosonicdb"  # Only from this dataset
        ... )
    """
    
    name: str
    labels: List[str]
    description: str = ""
    dataset_filter: str = ""
    color: Optional[str] = None
    
    def __post_init__(self):
        """Validate the sound class configuration."""
        # Normalize name (lowercase, replace spaces with underscores)
        self.name = self.name.lower().replace(" ", "_").replace("-", "_")
        
        # Ensure labels is a list
        if isinstance(self.labels, str):
            self.labels = [self.labels]
        
        # Validate we have at least one label
        if not self.labels:
            raise ValueError(f"SoundClass '{self.name}' must have at least one label")
        
        # Validate name doesn't conflict with reserved names
        reserved = {"background", "bg", "mix", "mixture", "residual"}
        if self.name in reserved:
            raise ValueError(
                f"'{self.name}' is reserved. Choose a different name for your sound class."
            )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SoundClass":
        """Create SoundClass from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        label_preview = self.labels[:3]
        if len(self.labels) > 3:
            label_preview.append(f"... (+{len(self.labels) - 3} more)")
        return f"SoundClass(name='{self.name}', labels={label_preview})"


@dataclass  
class SoundClassCollection:
    """A collection of sound classes with helper methods.
    
    This is used internally by Project to manage multiple sound classes.
    """
    
    classes: List[SoundClass] = field(default_factory=list)
    
    def add(self, sound_class: SoundClass) -> None:
        """Add a sound class to the collection.
        
        Args:
            sound_class: The SoundClass to add
            
        Raises:
            ValueError: If a class with the same name already exists
        """
        if self.get(sound_class.name) is not None:
            raise ValueError(
                f"Sound class '{sound_class.name}' already exists. "
                f"Use remove() first if you want to replace it."
            )
        self.classes.append(sound_class)
    
    def remove(self, name: str) -> bool:
        """Remove a sound class by name.
        
        Args:
            name: Name of the class to remove
            
        Returns:
            True if removed, False if not found
        """
        name = name.lower().replace(" ", "_").replace("-", "_")
        for i, cls in enumerate(self.classes):
            if cls.name == name:
                self.classes.pop(i)
                return True
        return False
    
    def get(self, name: str) -> Optional[SoundClass]:
        """Get a sound class by name.
        
        Args:
            name: Name of the class to find
            
        Returns:
            The SoundClass if found, None otherwise
        """
        name = name.lower().replace(" ", "_").replace("-", "_")
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None
    
    def names(self) -> List[str]:
        """Get list of all class names."""
        return [cls.name for cls in self.classes]
    
    def all_labels(self) -> List[str]:
        """Get flat list of all labels across all classes."""
        labels = []
        for cls in self.classes:
            labels.extend(cls.labels)
        return labels
    
    def to_target_classes(self) -> List[List[str]]:
        """Convert to target_classes format for TUSS config.
        
        Returns:
            List of label lists, one per class, in order.
        """
        return [cls.labels for cls in self.classes]
    
    def to_coi_prompts(self) -> List[str]:
        """Convert to coi_prompts format for TUSS config.
        
        Returns:
            List of class names, in order.
        """
        return [cls.name for cls in self.classes]
    
    def __len__(self) -> int:
        return len(self.classes)
    
    def __iter__(self):
        return iter(self.classes)
    
    def __getitem__(self, idx: int) -> SoundClass:
        return self.classes[idx]
    
    def to_list(self) -> List[dict]:
        """Convert to list of dicts for serialization."""
        return [cls.to_dict() for cls in self.classes]
    
    @classmethod
    def from_list(cls, data: List[dict]) -> "SoundClassCollection":
        """Create collection from list of dicts."""
        collection = cls()
        for item in data:
            collection.add(SoundClass.from_dict(item))
        return collection
