"""Configuration dataclasses for sudormrf training."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class COIClassConfig:
    """Configuration for one Class-of-Interest (COI) output head.

    Each entry in DataConfig.coi_classes defines exactly one COI branch in the
    separation model.

    Attributes:
        labels:  Semantic label strings used to match rows in the metadata
                 DataFrame (matched against the 'label' column).  Pure-COI
                 matching is applied: only recordings whose every label belongs
                 to this list are included.
        dataset: Optional dataset-name filter (matched against the 'dataset'
                 column with a case-insensitive substring match).  Leave empty
                 ("") to accept matching labels from any loaded dataset.
        name:    Human-readable name used in log output.  Defaults to the
                 first entry in `labels` when empty.

    Example (aircraft from aerosonicdb, birds from a custom loader):
        coi_classes:
          - labels: ["airplane", "Airplane", "plane"]
            dataset: "aerosonicdb"
            name: "aircraft"
          - labels: ["bird", "Bird", "birdsong"]
            dataset: "birds_dataset"
            name: "birds"
    """

    labels: List[str] = field(default_factory=list)
    dataset: str = ""  # "" = any dataset
    name: str = ""     # "" = use labels[0]


@dataclass
class DataConfig:
    df_path: str = "data/aircraft_data.csv"
    sample_rate: int = 16000
    segment_length: float = 5.0
    snr_range: List[float] = field(default_factory=lambda: [-5, 5])
    n_coi_classes: int = 1
    # list of semantic labels that define the class(es) of interest we want
    # the separation model to focus on.  This replaces the hard‑coded
    # `target_classes` list that used to live in `train.py`.  During training
    # the dataframe is filtered based on these labels and saved alongside the
    # checkpoint; during inference the same list can be recovered from the
    # checkpoint config.
    #
    # Single-class mode (n_coi_classes=1): set target_classes only.
    # Multi-class mode  (n_coi_classes>1): set coi_classes instead;
    #   target_classes is derived automatically as the union of all label lists.
    target_classes: List[str] = field(default_factory=list)
    # Per-class configuration for multi-class separation (n_coi_classes > 1).
    # Each entry maps to one output head (index 0, 1, ..., n_coi_classes-1).
    # Each entry specifies the semantic labels for that class AND the dataset
    # identifier to pull samples from, so different COI classes can come from
    # completely different datasets.  Background is always the last head and
    # needs no entry here.
    #
    # Leave empty ([]) for single-class mode (n_coi_classes=1).
    coi_classes: List[COIClassConfig] = field(default_factory=list)
    # Probability (0..1) of returning a background-only example during training
    background_only_prob: float = 0.0
    # How many non-COI files to mix together when creating a background-only example
    background_mix_n: int = 2
    # Augmentation multiplier: each COI sample is seen this many times per epoch
    # with different random augmentations (e.g., 3 = 3x more COI samples)
    augment_multiplier: int = 1
    # Probability (0..1) that each *other* COI class is added to a training mixture
    # (independently per class).  0.0 = disabled (single-class per sample).
    # Only meaningful when n_coi_classes > 1.
    multi_coi_prob: float = 0.0
    # WebDataset configuration
    # Set use_webdataset=True to load from tar shards instead of individual files
    use_webdataset: bool = False
    # Path to directory containing WebDataset tar shards
    # Required when use_webdataset=True
    webdataset_path: str = ""


@dataclass
class ModelConfig:
    type: str = "improved"
    out_channels: int = 256
    in_channels: int = 512
    num_blocks: int = 16
    upsampling_depth: int = 5
    enc_kernel_size: int = 41
    enc_num_basis: int = 512
    # COI separation head configuration
    num_head_conv_blocks: int = 0
    # If 0 (default): simple PReLU + Conv1d head (lightweight, baseline)
    # If > 0: enhanced head with N UConvBlocks per branch for class-specific features
    # Expanded dimension for UConvBlock internal processing (None = use in_channels)
    expanded_channels: Optional[int] = None


@dataclass
class TrainingConfig:
    batch_size: int = 16  # Increased from 4 for better GPU utilization
    grad_accum_steps: int = 1
    use_amp: bool = True
    num_epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 1e-2
    num_workers: int = 4  # Parallel data loading (adjust based on CPU cores)
    clip_grad_norm: float = 5.0
    patience: int = 15
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    compile_model: bool = True
    compile_backend: str = "inductor"
    # Class-weight for COI-focused loss (used by COILoss). Higher -> more
    # emphasis on COI reconstruction. Default 1.5 matches prior code.
    class_weight: float = 1.5
    # Optional auxiliary L1 weight on waveforms to stabilize early training.
    # Set to 0.0 to disable.
    aux_waveform_weight: float = 0.0
    # Whether to enable DataLoader pin_memory. Set True when training on GPU
    pin_memory: bool = True  # Enable for faster host->device transfer
    # DataLoader prefetch factor - how many batches to preload per worker
    prefetch_factor: int = 2
    # Keep worker processes alive between epochs
    persistent_workers: bool = True
    # Learning rate warmup: number of steps to linearly ramp up LR from 0 to lr
    # Helps prevent non-finite values in early training. Set 0 to disable.
    warmup_steps: int = 500
    # Validation frequency: run validation every N epochs (1 = every epoch)
    validate_every_n_epochs: int = 1
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "Config":
        data_dict = dict(cfg_dict.get("data", {}))
        # Deserialise nested COIClassConfig entries (YAML loads them as plain dicts)
        raw_coi_classes = data_dict.pop("coi_classes", []) or []
        data_cfg = DataConfig(**data_dict)
        data_cfg.coi_classes = [
            COIClassConfig(**c) if isinstance(c, dict) else c
            for c in raw_coi_classes
        ]
        model_cfg = ModelConfig(**cfg_dict.get("model", {}))
        training_cfg = TrainingConfig(**cfg_dict.get("training", {}))
        return cls(data=data_cfg, model=model_cfg, training=training_cfg)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return cls.from_dict(cfg_dict)

    def save(self, path: str | Path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self):
        return asdict(self)
