"""
Training presets for different use cases.

Presets provide sensible defaults for training parameters, hiding ML complexity
while allowing users to make simple choices like "quick test" vs "full training".
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum


class PresetName(str, Enum):
    """Available training presets."""
    QUICK = "quick"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    CUSTOM = "custom"


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
        }


# Pre-defined presets
PRESETS: Dict[str, TrainingPreset] = {
    "quick": TrainingPreset(
        name="quick",
        description="Quick test run. Use to verify pipeline works before full training.",
        epochs=20,
        patience=5,
        freeze_backbone=True,  # Faster, only train prompts
        batch_size=2,
        grad_accum_steps=4,
        warmup_steps=100,
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
        lr=3e-5,  # Slightly lower LR for more stable convergence
    ),
    
    "small_dataset": TrainingPreset(
        name="small_dataset",
        description="Optimized for small datasets (<500 samples). Uses frozen backbone.",
        epochs=150,
        patience=20,
        freeze_backbone=True,  # Prevent overfitting
        batch_size=2,
        grad_accum_steps=4,
        warmup_steps=200,
        lr=1e-4,  # Higher LR since only training prompts
        weight_decay=5e-3,  # Less regularization needed
    ),
    
    "large_dataset": TrainingPreset(
        name="large_dataset",
        description="Optimized for large datasets (>5000 samples). Full fine-tuning.",
        epochs=100,
        patience=15,
        freeze_backbone=False,
        batch_size=4,  # Can use larger batches
        grad_accum_steps=4,
        warmup_steps=500,
        lr=5e-5,
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
class DataSettings:
    """Data-related settings that users may want to tweak.
    
    These are settings that affect how training data is prepared,
    and are more understandable than ML hyperparameters.
    
    Attributes:
        sample_rate: Audio sample rate in Hz. Must match your audio files.
        segment_length: Duration of each training segment in seconds.
        snr_range: Signal-to-noise ratio range (dB) for mixing COI with background.
                  Use [-10, 10] for challenging conditions, [0, 20] for cleaner data.
        background_only_prob: Probability (0-1) that a training sample contains
                             only background (no COI). Helps model learn "nothing present".
        background_mix_n: How many background clips to mix together per sample.
        augment_multiplier: Each audio file is used this many times per epoch
                           with different augmentations.
    """
    
    sample_rate: int = 48000
    segment_length: float = 4.0
    snr_range: tuple = (-10, 10)
    background_only_prob: float = 0.3
    background_mix_n: int = 2
    augment_multiplier: int = 1
    
    def to_data_config(self) -> Dict[str, Any]:
        """Convert to data section of TUSS config."""
        return {
            "sample_rate": self.sample_rate,
            "segment_length": self.segment_length,
            "snr_range": list(self.snr_range),
            "background_only_prob": self.background_only_prob,
            "background_mix_n": self.background_mix_n,
            "augment_multiplier": self.augment_multiplier,
        }


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
