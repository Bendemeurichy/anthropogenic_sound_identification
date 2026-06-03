"""
Project management for sound separation pipelines.

A Project encapsulates everything needed to train and run inference for
a set of sound classes: the class definitions, training settings, data paths,
and trained model checkpoints.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import yaml

from .sound_class import SoundClass, SoundClassCollection
from .presets import (
    TrainingPreset,
    HardwareSettings,
    get_preset,
)


def _tuss_target_classes(collection: SoundClassCollection) -> List[List[str]]:
    """Convert collection to target_classes format for TUSS config."""
    return [cls.labels for cls in collection]


def _tuss_coi_prompts(collection: SoundClassCollection) -> List[str]:
    """Convert collection to coi_prompts format for TUSS config."""
    return [cls.name for cls in collection]


@dataclass
class ProjectSettings:
    """User-facing project settings.

    This is the simplified configuration that end-users interact with.
    It gets converted to a full TUSS training_config.yaml internally.

    Attributes:
        name: Project name (used for directory and checkpoint naming)
        description: Human-readable project description

        # Data settings
        data_path: Path to metadata CSV with audio file paths and labels
        sample_rate: Audio sample rate (Hz). Must match your audio files.
        segment_length: Training segment duration (seconds)
        snr_range: SNR range (dB) for mixing COI with background
        background_only_prob: Fraction of training samples with only background
        background_mix_n: Number of background clips to mix together
        augment_multiplier: Repeat multiplier for each audio file per epoch
        multi_coi_prob: Probability of mixing a second COI into the sample
        balance_classes: When True, sample each COI class equally (recommended)
        coi_class_multipliers: Per-class augmentation multipliers to balance
            imbalanced datasets (e.g. [5, 1] duplicates class-0 5x)

        # WebDataset (fast HPC loading from tar shards)
        use_webdataset: Use WebDataset tar shards instead of individual files
        webdataset_path: Path to WebDataset tar shards directory

        # Training settings
        preset: Training preset name ("quick", "balanced", "thorough", etc.)
        epochs: Override preset epochs (optional)
        patience: Override preset early stopping patience (optional)
        freeze_backbone: Override preset backbone freezing (optional)

        # Model settings
        pretrained_model: Path to pretrained TUSS model
        resume_from: Path to checkpoint to resume training from (optional)
        coi_prompt_init_from: Pretrained prompt to warm-start COI from
        bg_prompt_init_from: Pretrained prompt to warm-start background from

        # Inference settings
        enable_mask_recycling: Use mask recycling optimization for inference
        cache_size: Number of segments to cache
        similarity_threshold: Cosine similarity threshold for cache reuse
    """

    name: str = "my_project"
    description: str = ""

    # Data settings
    data_path: str = ""
    sample_rate: int = 48000
    segment_length: float = 6.0
    snr_range: tuple = (-10, 10)
    background_only_prob: float = 0.15
    background_mix_n: int = 2
    augment_multiplier: int = 1
    multi_coi_prob: float = 0.3
    balance_classes: bool = False
    coi_class_multipliers: Optional[List[float]] = None

    # WebDataset
    use_webdataset: bool = True
    webdataset_path: str = "$VSC_SCRATCH/webdataset_shards"

    # Training settings
    preset: str = "balanced"
    epochs: Optional[int] = None
    patience: Optional[int] = None
    freeze_backbone: Optional[bool] = None

    # Model settings
    pretrained_model: str = "base/pretrained_models/tuss.medium.2-4src"
    resume_from: str = ""
    coi_prompt_init_from: str = "sfx"
    bg_prompt_init_from: str = "sfxbg"

    # Inference settings
    enable_mask_recycling: bool = True
    cache_size: int = 5
    similarity_threshold: float = 0.85

    def get_training_preset(self) -> TrainingPreset:
        """Get the resolved training preset with any overrides applied."""
        preset = get_preset(self.preset)

        if self.epochs is not None:
            preset.epochs = self.epochs
        if self.patience is not None:
            preset.patience = self.patience
        if self.freeze_backbone is not None:
            preset.freeze_backbone = self.freeze_backbone

        return preset


@dataclass
class Project:
    """A sound separation project.

    Manages sound classes, training configuration, and trained models
    for a specific separation task.

    Example:
        >>> # Create a new project
        >>> project = Project.create("bird_study", base_dir="./projects")
        >>>
        >>> # Add sound classes
        >>> project.add_class(SoundClass(
        ...     name="bird",
        ...     labels=["bird", "birdsong"],
        ...     description="Bird vocalizations"
        ... ))
        >>> project.add_class(SoundClass(
        ...     name="airplane",
        ...     labels=["airplane", "aircraft"],
        ... ))
        >>>
        >>> # Configure data
        >>> project.settings.data_path = "path/to/metadata.csv"
        >>> project.settings.sample_rate = 48000
        >>>
        >>> # Train (uses balanced preset by default)
        >>> project.train()
        >>>
        >>> # Or train with quick preset for testing
        >>> project.train(preset="quick")
        >>>
        >>> # Run inference
        >>> sources = project.separate("recording.wav")
        >>> sources["bird"]  # Bird-only audio
    """

    settings: ProjectSettings
    classes: SoundClassCollection = field(default_factory=SoundClassCollection)
    project_dir: Optional[Path] = None

    # Internal state
    _checkpoint_path: Optional[Path] = None
    _is_trained: bool = False
    _created_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        name: str,
        base_dir: Union[str, Path] = "./projects",
        description: str = "",
    ) -> "Project":
        """Create a new project.

        Creates a project directory structure:
            {base_dir}/{name}/
                project.yaml      # Project configuration
                checkpoints/      # Trained model checkpoints
                logs/             # Training logs

        Args:
            name: Project name (used for directory naming)
            base_dir: Parent directory for projects
            description: Human-readable project description

        Returns:
            New Project instance
        """
        base_dir = Path(base_dir)
        project_dir = base_dir / name

        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "checkpoints").mkdir(exist_ok=True)
        (project_dir / "logs").mkdir(exist_ok=True)

        settings = ProjectSettings(name=name, description=description)
        project = cls(settings=settings, project_dir=project_dir)

        project.save()

        return project

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Project":
        """Load a project from disk.

        Args:
            path: Path to project directory or project.yaml file

        Returns:
            Loaded Project instance
        """
        path = Path(path)

        if path.is_file():
            config_path = path
            project_dir = path.parent
        else:
            project_dir = path
            config_path = path / "project.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        metadata = data.get("metadata", {})

        settings_data = data.get("settings", {})
        if "snr_range" in settings_data and isinstance(settings_data["snr_range"], list):
            settings_data["snr_range"] = tuple(settings_data["snr_range"])
        settings = ProjectSettings(**settings_data)

        classes_data = data.get("classes", [])
        classes = SoundClassCollection.from_list(classes_data)

        created_str = metadata.get("created")
        created_at = datetime.fromisoformat(created_str) if created_str else None

        project = cls(
            settings=settings,
            classes=classes,
            project_dir=project_dir,
            _created_at=created_at,
        )

        project._find_latest_checkpoint()

        return project

    def save(self) -> None:
        """Save project configuration to disk."""
        if self.project_dir is None:
            raise ValueError("Project has no directory. Use Project.create() to create one.")

        config_path = self.project_dir / "project.yaml"

        if self._created_at is None:
            self._created_at = datetime.now()

        settings_dict = asdict(self.settings)
        settings_dict["snr_range"] = list(self.settings.snr_range)

        data = {
            "settings": settings_dict,
            "classes": self.classes.to_list(),
            "metadata": {
                "created": self._created_at.isoformat(),
                "version": "2",
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def add_class(self, sound_class: SoundClass) -> None:
        """Add a sound class to the project.

        Args:
            sound_class: SoundClass to add

        Raises:
            ValueError: If class with same name already exists
        """
        self.classes.add(sound_class)
        self.save()

    def remove_class(self, name: str) -> bool:
        """Remove a sound class from the project.

        Args:
            name: Name of class to remove

        Returns:
            True if removed, False if not found
        """
        result = self.classes.remove(name)
        if result:
            self.save()
        return result

    def get_class(self, name: str) -> Optional[SoundClass]:
        """Get a sound class by name."""
        return self.classes.get(name)

    def list_classes(self) -> List[str]:
        """List all sound class names."""
        return self.classes.names()

    def generate_training_config(self) -> Dict[str, Any]:
        """Generate full TUSS training_config.yaml content.

        Converts the simplified project settings to the full config format
        expected by the TUSS training script.

        Returns:
            Dictionary that can be written as training_config.yaml
        """
        preset = self.settings.get_training_preset()
        hardware = HardwareSettings().resolve()

        return {
            "data": self._build_data_config(),
            "model": self._build_model_config(preset),
            "training": self._build_training_config(preset, hardware),
        }

    def _build_data_config(self) -> Dict[str, Any]:
        config = {
            "df_path": self.settings.data_path,
            "sample_rate": self.settings.sample_rate,
            "segment_length": self.settings.segment_length,
            "snr_range": list(self.settings.snr_range),
            "target_classes": _tuss_target_classes(self.classes),
            "background_only_prob": self.settings.background_only_prob,
            "background_mix_n": self.settings.background_mix_n,
            "augment_multiplier": self.settings.augment_multiplier,
            "multi_coi_prob": self.settings.multi_coi_prob,
            "balance_classes": self.settings.balance_classes,
        }
        if self.settings.coi_class_multipliers is not None:
            config["coi_class_multipliers"] = self.settings.coi_class_multipliers
        if self.settings.use_webdataset:
            config["use_webdataset"] = self.settings.use_webdataset
            config["webdataset_path"] = self.settings.webdataset_path
        return config

    def _build_model_config(self, preset: TrainingPreset) -> Dict[str, Any]:
        return {
            "pretrained_path": self.settings.pretrained_model,
            "coi_prompts": _tuss_coi_prompts(self.classes),
            "bg_prompt": "background",
            "coi_prompt_init_from": self.settings.coi_prompt_init_from,
            "bg_prompt_init_from": self.settings.bg_prompt_init_from,
            "freeze_backbone": preset.freeze_backbone,
            # Architecture defaults (only used if pretrained_path is null)
            "encoder_name": "stft",
            "encoder_conf": {},
            "decoder_name": "stft",
            "decoder_conf": {},
            "separator_name": "tuss",
            "separator_conf": {},
            "css_conf": {},
            "variance_normalization": True,
        }

    def _build_training_config(self, preset: TrainingPreset, hardware: HardwareSettings) -> Dict[str, Any]:
        return {
            **preset.to_training_config(),
            **hardware.to_training_config(),
            "checkpoint_dir": str(self.project_dir / "checkpoints") if self.project_dir else "checkpoints",
            "validate_every_n_epochs": 1,
            "resume_from": self.settings.resume_from,
        }

    def write_training_config(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Write the training config to a YAML file.

        Args:
            path: Output path. Defaults to {project_dir}/training_config.yaml

        Returns:
            Path to written config file
        """
        if path is None:
            if self.project_dir is None:
                raise ValueError("No output path specified and project has no directory")
            path = self.project_dir / "training_config.yaml"
        else:
            path = Path(path)

        config = self.generate_training_config()

        header = """# =============================================================================
# TUSS Training Configuration
# Generated by Project: {name}
# Generated at: {timestamp}
# =============================================================================
#
# This file was auto-generated from project.yaml settings.
# To modify training parameters, edit project.yaml and regenerate,
# or edit this file directly for one-off changes.
#
# =============================================================================

""".format(name=self.settings.name, timestamp=datetime.now().isoformat())

        with open(path, "w") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return path

    def validate(self) -> List[str]:
        """Validate project configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if len(self.classes) == 0:
            errors.append("No sound classes defined. Add at least one class with add_class().")

        if self.settings.use_webdataset:
            if not self.settings.webdataset_path:
                errors.append("WebDataset mode enabled but no webdataset_path set.")
        else:
            if not self.settings.data_path:
                errors.append("No data path specified. Set settings.data_path or enable use_webdataset.")
            elif not Path(self.settings.data_path).exists():
                errors.append(f"Data path does not exist: {self.settings.data_path}")

        if self.settings.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            errors.append(
                f"Unusual sample rate: {self.settings.sample_rate}. "
                f"Common values are 16000, 44100, or 48000."
            )

        snr_min, snr_max = self.settings.snr_range
        if snr_min > snr_max:
            errors.append(f"Invalid SNR range: min ({snr_min}) > max ({snr_max})")

        return errors

    def _find_latest_checkpoint(self) -> None:
        """Find the latest checkpoint in the project directory."""
        if self.project_dir is None:
            return

        checkpoints_dir = self.project_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return

        checkpoint_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )

        for ckpt_dir in checkpoint_dirs:
            best_model = ckpt_dir / "best_model.pt"
            if best_model.exists():
                self._checkpoint_path = ckpt_dir
                self._is_trained = True
                return

    @property
    def is_trained(self) -> bool:
        """Check if project has a trained model."""
        return self._is_trained

    @property
    def checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint directory."""
        return self._checkpoint_path

    def train(
        self,
        preset: Optional[str] = None,
        epochs: Optional[int] = None,
        device: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """Train the separation model.

        Args:
            preset: Override project preset ("quick", "balanced", "thorough", etc.)
            epochs: Override number of epochs
            device: Override device ("cuda", "cuda:0", "cpu")
            dry_run: If True, only generate config without training

        Returns:
            Path to checkpoint directory if training completed, None if dry_run

        Raises:
            ValueError: If project configuration is invalid
        """
        if preset is not None:
            self.settings.preset = preset
        if epochs is not None:
            self.settings.epochs = epochs

        errors = self.validate()
        if errors:
            raise ValueError("Invalid project configuration:\n  - " + "\n  - ".join(errors))

        config_path = self.write_training_config()

        if dry_run:
            print(f"Training config written to: {config_path}")
            print("Dry run - not starting training.")
            return None

        from .runner import run_training

        checkpoint_path = run_training(
            config_path=config_path,
            device=device,
        )

        self._checkpoint_path = checkpoint_path
        self._is_trained = True

        return checkpoint_path

    def _require_trained(self) -> None:
        """Raise ValueError if the project has no trained model."""
        if not self._is_trained or self._checkpoint_path is None:
            raise ValueError(
                "Project has no trained model. Run train() first, "
                "or set settings.resume_from to an existing checkpoint."
            )

    def separate(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Separate audio into sound classes.

        Args:
            audio_path: Path to audio file to process
            output_dir: Optional directory to save separated audio files
            classes: Optional list of class names to pass as COI prompts

        Returns:
            Dictionary mapping prompt names to separated audio tensors

        Raises:
            ValueError: If project has no trained model
        """
        self._require_trained()

        from .runner import run_inference

        return run_inference(
            checkpoint_path=self._checkpoint_path,
            audio_path=audio_path,
            output_dir=output_dir,
            coi_prompt=classes or self.classes.names(),
            bg_prompt="background",
            enable_mask_recycling=self.settings.enable_mask_recycling,
            cache_size=self.settings.cache_size,
            similarity_threshold=self.settings.similarity_threshold,
        )

    def get_pipeline(self):
        """Get a SeparationPipeline for this project.

        Returns a configured SeparationPipeline that can be used for
        batch inference or integration with other code.

        Returns:
            SeparationPipeline instance

        Raises:
            ValueError: If project has no trained model
        """
        self._require_trained()

        from .runner import create_pipeline

        return create_pipeline(
            checkpoint_path=self._checkpoint_path,
            coi_prompt=self.classes.names(),
            bg_prompt="background",
            enable_mask_recycling=self.settings.enable_mask_recycling,
            cache_size=self.settings.cache_size,
            similarity_threshold=self.settings.similarity_threshold,
        )

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "not trained"
        return (
            f"Project(name='{self.settings.name}', "
            f"classes={self.classes.names()}, "
            f"status={status})"
        )
