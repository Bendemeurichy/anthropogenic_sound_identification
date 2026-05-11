"""Configuration settings for AST training"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class TrainingConfig:
    """Training configuration parameters for AST plane classifier"""

    filename_column: str = "filename"
    start_time_column: str = "start_time"
    end_time_column: str = "end_time"
    label_column: str = "label"
    split_column: str = "split"
    batch_size: int = 16
    sample_rate: int = 16000  # AST uses 16kHz
    audio_duration: float = 10.0
    split_long: bool = True
    min_clip_length: float = 0.5
    shuffle_buffer: int = 10000
    num_workers: int = 1
    pin_memory: bool = True

    # Training parameters - Phase 1 (frozen backbone)
    phase1_epochs: int = 30
    phase1_lr: float = 1e-3
    phase1_patience: int = 10
    phase1_reduce_lr_patience: int = 5

    # Training parameters - Phase 2 (fine-tuning)
    phase2_epochs: int = 20
    phase2_lr: float = 1e-5
    phase2_patience: int = 8
    phase2_reduce_lr_patience: int = 4

    weight_decay: float = 1e-5
    beta_1: float = 0.9
    beta_2: float = 0.999
    gradient_clip_norm: float = 1.0

    label_smoothing: float = 0.0
    dropout_rate_1: float = 0.3
    dropout_rate_2: float = 0.2
    dropout_rate_3: float = 0.1

    use_augmentation: bool = True
    aug_time_stretch_prob: float = 0.5
    aug_time_stretch_range: tuple = (0.95, 1.05)
    aug_noise_prob: float = 0.5
    aug_noise_stddev: float = 0.005
    aug_gain_prob: float = 0.5
    aug_gain_range: tuple = (0.7, 1.3)

    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    hidden_units: List[int] = field(default_factory=lambda: [512, 256, 128])

    bootstrap_enabled: bool = True
    bootstrap_n_iterations: int = 100
    bootstrap_confidence_level: float = 0.95
    bootstrap_log_frequency: int = 1

    random_seed: int = 42

    class_weight_mode: Optional[str] = None
    manual_class_weights: Optional[Dict[int, float]] = None

    device: str = "cuda"


@dataclass
class ModelConfig:
    """Model architecture configuration for AST classifier"""

    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    sample_rate: int = 16000
    embedding_dim: int = 768  # AST standard embedding dimension

    hidden_units: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"
    use_batch_norm: bool = True
    dropout_rates: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.1])

    def __post_init__(self):
        if len(self.dropout_rates) != len(self.hidden_units):
            if len(self.dropout_rates) < len(self.hidden_units):
                self.dropout_rates = self.dropout_rates + [0.1] * (
                    len(self.hidden_units) - len(self.dropout_rates)
                )
            else:
                self.dropout_rates = self.dropout_rates[: len(self.hidden_units)]
