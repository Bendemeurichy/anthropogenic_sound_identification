"""Configuration settings for training"""

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    # Data parameters
    filename_column: str = "filename"
    start_time_column: str = "start_time"
    end_time_column: str = "end_time"
    label_column: str = "label"
    split_column: str = "split"
    batch_size: int = 32
    sample_rate: int = 16000  # YAMNet requirement
    audio_duration: float = 5.0  # Duration in seconds for each audio clip
    split_long: bool = True  # Split long annotations into multiple clips
    min_clip_length: float = 0.5  # Minimum clip length in seconds
    min_audio_duration: float = 0.5  # Minimum duration to keep clip
    shuffle_buffer: int = 10000

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

    # Optimizer parameters
    weight_decay: float = 1e-5
    beta_1: float = 0.9
    beta_2: float = 0.999
    clipnorm: float = 1.0

    # Regularization
    label_smoothing: float = 0.1
    l2_reg: float = 1e-4
    dropout_rate_1: float = 0.3
    dropout_rate_2: float = 0.2
    dropout_rate_3: float = 0.1

    # Augmentation
    use_augmentation: bool = True
    aug_time_stretch_prob: float = 0.5
    aug_time_stretch_range: tuple = (0.8, 1.2)
    aug_noise_prob: float = 0.5
    aug_noise_stddev: float = 0.005
    aug_gain_prob: float = 0.5
    aug_gain_range: tuple = (0.7, 1.3)

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Model architecture
    hidden_units: Optional[List[int]] = None

    # Bootstrap CI for validation metrics
    bootstrap_enabled: bool = True
    bootstrap_n_iterations: int = 1000
    bootstrap_confidence_level: float = 0.95
    bootstrap_log_frequency: int = 1  # Log bootstrap CI every N epochs

    # Random seed for reproducibility
    random_seed: int = 42

    # Class weight mode: 'balanced', 'sqrt_balanced', 'manual', or None
    class_weight_mode: Optional[str] = "sqrt_balanced"
    # Manual class weights (used only if class_weight_mode='manual')
    manual_class_weights: Optional[Dict[int, float]] = None

    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [512, 256, 128]


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    yamnet_url: str = "https://tfhub.dev/google/yamnet/1"
    embedding_dim: int = 1024  # YAMNet output dimension
    hidden_units: Optional[List[int]] = None
    activation: str = "swish"
    use_batch_norm: bool = True
    dropout_rates: Optional[List[float]] = None
    l2_reg: float = 1e-4

    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [512, 256, 128]
        if self.dropout_rates is None:
            self.dropout_rates = [0.3, 0.2, 0.1]
