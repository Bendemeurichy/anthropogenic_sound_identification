"""
Configuration module for TUSS model fine-tuning.

Provides device resolution, data/model/training configuration dataclasses
with YAML serialization, and validation logic.
"""

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import yaml


# =============================================================================
# Device resolution
# =============================================================================


def resolve_device(device: str | int) -> str:
    """Return a concrete device string from a flexible specification.

    Accepted forms
    ──────────────
    "cuda"      → "cuda:<index of best GPU>" (or "cpu" if none available)
    "cuda:N"    → validated, falls back to "cpu" if GPU N is absent
    N  (int)    → "cuda:N"  (e.g. 0, 1, 2 …)
    "cpu"       → "cpu"

    The resolved string is always safe to pass to ``tensor.to(device)``.
    """
    if isinstance(device, int):
        device = f"cuda:{device}"

    device = str(device).strip().lower()

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            print("No CUDA device found – falling back to CPU")
            return "cpu"
        idx = torch.cuda.current_device()
        return f"cuda:{idx}"

    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            print("No CUDA device found – falling back to CPU")
            return "cpu"
        try:
            idx = int(device.split(":")[1])
        except ValueError:
            print(f"Invalid device string '{device}' – falling back to cuda:0")
            idx = 0
        n_gpus = torch.cuda.device_count()
        if idx >= n_gpus:
            print(
                f"GPU {idx} requested but only {n_gpus} GPU(s) available – "
                f"falling back to cuda:0"
            )
            idx = 0
        return f"cuda:{idx}"

    print(f"Unrecognised device '{device}' – falling back to CPU")
    return "cpu"


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class DataConfig:
    df_path: str = "data/aircraft_data.csv"
    sample_rate: int = 48000
    segment_length: float = 6.0
    snr_range: list = field(default_factory=lambda: [-5, 5])
    # Nested list: target_classes[i] is a list of label strings mapping to
    # coi_prompts[i].  A flat list is also accepted (treated as one class).
    target_classes: list = field(default_factory=list)
    background_only_prob: float = 0.15
    background_mix_n: int = 2
    augment_multiplier: int = 2
    multi_coi_prob: float = 0.3
    balance_classes: bool = False
    coi_class_multipliers: Optional[list] = None  # Per-class augmentation multipliers
    # WebDataset configuration
    use_webdataset: bool = False
    webdataset_path: str = ""


@dataclass
class ModelConfig:
    pretrained_path: str = "base/pretrained_models/tuss.medium.2-4src"
    coi_prompts: list = field(default_factory=lambda: ["airplane", "train", "bird"])
    bg_prompt: str = "background"
    coi_prompt_init_from: str = "sfx"
    bg_prompt_init_from: str = "sfxbg"
    freeze_backbone: bool = False
    # From-scratch architecture (used when pretrained_path is null)
    encoder_name: str = "stft"
    encoder_conf: dict = field(default_factory=dict)
    decoder_name: str = "stft"
    decoder_conf: dict = field(default_factory=dict)
    separator_name: str = "tuss"
    separator_conf: dict = field(default_factory=dict)
    css_conf: dict = field(default_factory=dict)
    variance_normalization: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 2
    grad_accum_steps: int = 8
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" (recommended, matches pretrained) or "fp16"
    num_epochs: int = 200
    lr: float = 5e-5
    weight_decay: float = 1e-2
    num_workers: int = 8  # Increased from 4 for faster data loading
    pin_memory: bool = True
    clip_grad_norm: float = 5.0
    patience: int = 30
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    coi_weight: float = 1.5
    snr_max: int = 30
    zero_ref_loss_weight: float = 0.1
    warmup_steps: int = 300
    validate_every_n_epochs: int = 1
    resume_from: str = ""  # path to checkpoint .pt file to resume training
    seed: int = 42
    existing_prompt_lr_multiplier: float = (
        0.1  # LR multiplier for prompts that exist in checkpoint
    )
    stabilization_epochs: int = 0  # Number of epochs to freeze old prompts and backbone
    backbone_lr_multiplier: float = 0.05  # LR multiplier for backbone when unfreezing
    # Prompt variability settings (similar to base TUSS training)
    variable_prompts: bool = False  # Enable variable prompt configurations during training
    prompt_dropout_prob: float = 0.5  # Probability of dropping each COI prompt during training
    min_coi_prompts: int = 0  # Minimum number of COI prompts per sample (0 = allow background-only)
    # ReduceLROnPlateau scheduler settings
    scheduler_patience: int = 5    # epochs without improvement before reducing LR
    scheduler_factor: float = 0.5  # multiplicative factor when reducing LR
    scheduler_min_lr: float = 1e-7  # floor LR for all param groups
    # GPU augmentation settings (10-100x faster than CPU)
    use_gpu_augmentations: bool = True  # Apply augmentations on GPU instead of CPU
    gpu_aug_time_stretch_prob: float = 0.5
    gpu_aug_gain_prob: float = 0.7
    gpu_aug_noise_prob: float = 0.4
    gpu_aug_shift_prob: float = 0.5
    gpu_aug_lpf_prob: float = 0.3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        data_cfg = DataConfig(**raw.get("data", {}))
        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        config = cls(data=data_cfg, model=model_cfg, training=training_cfg)
        config.validate()
        return config

    def validate(self):
        """Validate configuration consistency."""
        # Check that target_classes and coi_prompts have matching lengths
        target_classes = self.data.target_classes
        coi_prompts = self.model.coi_prompts
        
        if not target_classes:
            raise ValueError(
                "Configuration error: data.target_classes is empty.\n"
                "Add class label lists to training_config.yaml under data.target_classes"
            )
        
        if not coi_prompts:
            raise ValueError(
                "Configuration error: model.coi_prompts is empty.\n"
                "Add prompt names to training_config.yaml under model.coi_prompts"
            )
        
        # Normalize target_classes to list-of-lists for validation
        # (single-class case: flat list is allowed)
        if isinstance(target_classes[0], str):
            target_classes_normalized = [target_classes]
        else:
            target_classes_normalized = target_classes
        
        n_coi_prompts = len(coi_prompts)
        n_target_classes = len(target_classes_normalized)
        
        if n_target_classes != n_coi_prompts:
            raise ValueError(
                f"Configuration error: Length mismatch between target_classes and coi_prompts.\n"
                f"  data.target_classes has {n_target_classes} groups: {target_classes_normalized}\n"
                f"  model.coi_prompts has {n_coi_prompts} entries: {coi_prompts}\n"
                f"These must match - one target_classes group per coi_prompt.\n"
                f"Check training_config.yaml and ensure both lists have the same length."
            )


    def save(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self):
        return asdict(self)
