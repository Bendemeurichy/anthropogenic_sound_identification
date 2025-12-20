"""Configuration dataclasses for sudormrf training."""

import yaml
from dataclasses import dataclass, field, asdict
from typing import List
from pathlib import Path


@dataclass
class DataConfig:
    df_path: str = "data/aircraft_data.csv"
    sample_rate: int = 16000
    segment_length: float = 5.0
    snr_range: List[float] = field(default_factory=lambda: [-5, 5])
    n_coi_classes: int = 1
    # Probability (0..1) of returning a background-only example during training
    background_only_prob: float = 0.0
    # How many non-COI files to mix together when creating a background-only example
    background_mix_n: int = 2
    # Augmentation multiplier: each COI sample is seen this many times per epoch
    # with different random augmentations (e.g., 3 = 3x more COI samples)
    augment_multiplier: int = 1


@dataclass
class ModelConfig:
    type: str = "improved"
    out_channels: int = 256
    in_channels: int = 512
    num_blocks: int = 16
    upsampling_depth: int = 5
    enc_kernel_size: int = 21
    enc_num_basis: int = 512


@dataclass
class TrainingConfig:
    batch_size: int = 16  # Increased from 4 for better GPU utilization
    grad_accum_steps: int = 1
    use_amp: bool = True
    num_epochs: int = 50
    lr: float = 0.001
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
        data_cfg = DataConfig(**cfg_dict.get("data", {}))
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
