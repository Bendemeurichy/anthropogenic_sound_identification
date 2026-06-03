"""
Training presets for different use cases.

Presets provide sensible defaults for training parameters, hiding ML complexity
while allowing users to make simple choices like "quick test" vs "full training".
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrainingPreset:
    """Training configuration preset.

    Presets bundle ML hyperparameters that work well together, so users
    don't need to understand learning rates, loss weights, etc.

    Attributes:
        name: Preset identifier
        description: Human-readable description
        epochs: Maximum training epochs
        patience: Early stopping patience (epochs without improvement)
        freeze_backbone: If True, only train prompt vectors (faster, less overfitting)
        batch_size: Samples per training step (auto-scaled based on GPU memory)

        # Advanced settings (hidden from basic users)
        lr: Learning rate
        weight_decay: L2 regularization
        warmup_steps: LR warmup steps
        grad_accum_steps: Gradient accumulation steps
        coi_weight: Loss weight for COI classes
        zero_ref_loss_weight: Penalty for predicting energy when target is silent
        clip_grad_norm: Gradient clipping threshold
        use_amp: Use mixed precision training
        amp_dtype: AMP dtype ("bf16" or "fp16")
        snr_max: SNR clipping threshold for loss

        # Scheduler settings
        scheduler_patience: ReduceLROnPlateau patience (epochs)
        scheduler_factor: Multiplicative LR reduction factor
        scheduler_min_lr: Floor LR

        # Prompt extension / differential LR
        stabilization_epochs: Freeze old prompts + backbone for N epochs
        backbone_lr_multiplier: LR multiplier for backbone relative to base LR
        existing_prompt_lr_multiplier: LR multiplier for existing prompts

        # Prompt variability (mimics base TUSS training)
        variable_prompts: Enable random prompt dropout during training
        prompt_dropout_prob: Probability of dropping each COI prompt
        min_coi_prompts: Minimum COI prompts per sample

        # GPU augmentations (10-100x faster than CPU)
        use_gpu_augmentations: Apply augmentations on GPU
        gpu_aug_time_stretch_prob: Time-stretch augmentation probability
        gpu_aug_gain_prob: Gain augmentation probability
        gpu_aug_noise_prob: Noise augmentation probability
        gpu_aug_shift_prob: Time-shift augmentation probability
        gpu_aug_lpf_prob: Low-pass filter augmentation probability
    """

    name: str
    description: str

    # User-facing parameters
    epochs: int = 100
    patience: int = 20
    freeze_backbone: bool = False
    batch_size: int = 2

    # Advanced parameters (with sensible defaults)
    lr: float = 5e-5
    weight_decay: float = 1e-2
    warmup_steps: int = 300
    grad_accum_steps: int = 8
    coi_weight: float = 2.0
    zero_ref_loss_weight: float = 0.1
    clip_grad_norm: float = 5.0
    use_amp: bool = True
    amp_dtype: str = "bf16"
    snr_max: float = 30.0
    existing_prompt_lr_multiplier: float = 0.1

    # Scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7

    # Differential LR / prompt extension
    stabilization_epochs: int = 0
    backbone_lr_multiplier: float = 0.05

    # Prompt variability
    variable_prompts: bool = False
    prompt_dropout_prob: float = 0.5
    min_coi_prompts: int = 0

    # GPU augmentations
    use_gpu_augmentations: bool = True
    gpu_aug_time_stretch_prob: float = 0.5
    gpu_aug_gain_prob: float = 0.7
    gpu_aug_noise_prob: float = 0.4
    gpu_aug_shift_prob: float = 0.5
    gpu_aug_lpf_prob: float = 0.3

    def to_training_config(self) -> Dict[str, Any]:
        """Convert to training section of TUSS config."""
        return {
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "num_epochs": self.epochs,
            "patience": self.patience,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "clip_grad_norm": self.clip_grad_norm,
            "existing_prompt_lr_multiplier": self.existing_prompt_lr_multiplier,
            "coi_weight": self.coi_weight,
            "zero_ref_loss_weight": self.zero_ref_loss_weight,
            "snr_max": self.snr_max,
            # Scheduler
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_min_lr": self.scheduler_min_lr,
            # Differential LR
            "stabilization_epochs": self.stabilization_epochs,
            "backbone_lr_multiplier": self.backbone_lr_multiplier,
            # Prompt variability
            "variable_prompts": self.variable_prompts,
            "prompt_dropout_prob": self.prompt_dropout_prob,
            "min_coi_prompts": self.min_coi_prompts,
            # GPU augmentations
            "use_gpu_augmentations": self.use_gpu_augmentations,
            "gpu_aug_time_stretch_prob": self.gpu_aug_time_stretch_prob,
            "gpu_aug_gain_prob": self.gpu_aug_gain_prob,
            "gpu_aug_noise_prob": self.gpu_aug_noise_prob,
            "gpu_aug_shift_prob": self.gpu_aug_shift_prob,
            "gpu_aug_lpf_prob": self.gpu_aug_lpf_prob,
        }


# Pre-defined presets
PRESETS: Dict[str, TrainingPreset] = {
    "quick": TrainingPreset(
        name="quick",
        description="Quick test run. Use to verify pipeline works before full training.",
        epochs=20,
        patience=5,
        freeze_backbone=True,
        batch_size=2,
        grad_accum_steps=4,
        warmup_steps=100,
        variable_prompts=False,
    ),

    "balanced": TrainingPreset(
        name="balanced",
        description="Balanced training. Good tradeoff between speed and quality.",
        epochs=100,
        patience=15,
        freeze_backbone=False,
        batch_size=2,
        grad_accum_steps=8,
        warmup_steps=300,
        variable_prompts=True,
        prompt_dropout_prob=0.15,
        min_coi_prompts=1,
    ),

    "thorough": TrainingPreset(
        name="thorough",
        description="Thorough training. Best quality, longer training time.",
        epochs=200,
        patience=25,
        freeze_backbone=False,
        batch_size=2,
        grad_accum_steps=8,
        warmup_steps=500,
        lr=3e-5,
        variable_prompts=True,
        prompt_dropout_prob=0.15,
        min_coi_prompts=1,
    ),

    "small_dataset": TrainingPreset(
        name="small_dataset",
        description="Optimized for small datasets (<500 samples). Uses frozen backbone.",
        epochs=150,
        patience=20,
        freeze_backbone=True,
        batch_size=2,
        grad_accum_steps=4,
        warmup_steps=200,
        lr=1e-4,
        weight_decay=5e-3,
        variable_prompts=False,
    ),

    "large_dataset": TrainingPreset(
        name="large_dataset",
        description="Optimized for large datasets (>5000 samples). Full fine-tuning.",
        epochs=100,
        patience=15,
        freeze_backbone=False,
        batch_size=4,
        grad_accum_steps=4,
        warmup_steps=500,
        lr=5e-5,
        variable_prompts=True,
        prompt_dropout_prob=0.15,
        min_coi_prompts=1,
    ),
}


def get_preset(name: str) -> TrainingPreset:
    """Get a preset by name.

    Args:
        name: Preset name ("quick", "balanced", "thorough", "small_dataset", "large_dataset")

    Returns:
        TrainingPreset with configured parameters

    Raises:
        ValueError: If preset name is not recognized
    """
    name = name.lower()
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions.

    Returns:
        Dict mapping preset names to descriptions
    """
    return {name: preset.description for name, preset in PRESETS.items()}


@dataclass
class HardwareSettings:
    """Hardware and system settings.

    These are auto-detected when possible, but can be overridden.

    Attributes:
        device: PyTorch device ("cuda", "cuda:0", "cpu"). Auto-detected if None.
        num_workers: DataLoader workers. Auto-detected based on CPU cores if None.
        pin_memory: Pin memory for faster GPU transfer. Auto-enabled for CUDA.
        seed: Random seed for reproducibility.
    """

    device: Optional[str] = None
    num_workers: Optional[int] = None
    pin_memory: bool = True
    seed: int = 42

    def resolve(self) -> "HardwareSettings":
        """Resolve auto-detected settings.

        Returns:
            New HardwareSettings with auto-detected values filled in.
        """
        import torch
        import os

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        num_workers = self.num_workers
        if num_workers is None:
            cpu_count = os.cpu_count() or 4
            num_workers = min(4, cpu_count // 2)

        pin_memory = self.pin_memory and device.startswith("cuda")

        return HardwareSettings(
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=self.seed,
        )

    def to_training_config(self) -> Dict[str, Any]:
        """Convert to training section hardware settings."""
        resolved = self.resolve()
        return {
            "device": resolved.device,
            "num_workers": resolved.num_workers,
            "pin_memory": resolved.pin_memory,
            "seed": resolved.seed,
        }
