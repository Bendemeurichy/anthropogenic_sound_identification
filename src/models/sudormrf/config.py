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
    segment_stride: float = 4.0
    snr_range: List[float] = field(default_factory=lambda: [-5, 5])
    n_coi_classes: int = 1


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
    batch_size: int = 4
    grad_accum_steps: int = 1
    use_amp: bool = True
    num_epochs: int = 50
    lr: float = 0.001
    num_workers: int = 4
    clip_grad_norm: float = 5.0
    patience: int = 15
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    compile_model: bool = False  # torch.compile can be slow on WSL with inductor
    compile_backend: str = "inductor"  # Options: 'inductor', 'eager', 'aot_eager'


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
