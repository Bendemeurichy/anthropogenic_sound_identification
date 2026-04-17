"""Configuration settings for PANN training"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class TrainingConfig:
    """Training configuration parameters for PANN plane classifier"""

    # Data parameters
    filename_column: str = "filename"
    start_time_column: str = "start_time"
    end_time_column: str = "end_time"
    label_column: str = "label"
    split_column: str = "split"
    batch_size: int = 16  # Smaller than YAMNet due to larger embeddings (2048 vs 1024)
    sample_rate: int = 32000  # PANN native sample rate (vs YAMNet's 16kHz)
    audio_duration: float = 10.0  # PANN works better with longer clips (vs YAMNet's 5s)
    split_long: bool = True  # Split long annotations into multiple clips
    min_clip_length: float = 0.5  # Minimum clip length in seconds
    shuffle_buffer: int = (
        10000  # Not used in PyTorch DataLoader, kept for compatibility
    )
    num_workers: int = 1  # Number of data loading workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer

    # Training parameters - Phase 1 (frozen backbone)
    phase1_epochs: int = 30
    phase1_lr: float = 1e-3  # Higher LR for training classifier head from scratch
    phase1_patience: int = 10
    phase1_reduce_lr_patience: int = 5

    # Training parameters - Phase 2 (fine-tuning)
    phase2_epochs: int = 20
    phase2_lr: float = 1e-5  # Much lower LR for fine-tuning pretrained CNN14
    phase2_patience: int = 8
    phase2_reduce_lr_patience: int = 4

    # Optimizer parameters
    weight_decay: float = 1e-5
    beta_1: float = 0.9
    beta_2: float = 0.999
    gradient_clip_norm: float = 1.0

    # Regularization
    label_smoothing: float = 0.0  # Label smoothing for BCE loss
    dropout_rate_1: float = 0.3  # First dense layer dropout
    dropout_rate_2: float = 0.2  # Second dense layer dropout
    dropout_rate_3: float = 0.1  # Third dense layer dropout

    # Augmentation (training only)
    use_augmentation: bool = True
    aug_time_stretch_prob: float = 0.5
    aug_time_stretch_range: tuple = (0.95, 1.05)
    aug_noise_prob: float = 0.5
    aug_noise_stddev: float = 0.005
    aug_gain_prob: float = 0.5
    aug_gain_range: tuple = (0.7, 1.3)

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Model architecture (classifier head)
    hidden_units: List[int] = field(default_factory=lambda: [512, 256, 128])

    # Bootstrap CI for validation metrics
    bootstrap_enabled: bool = True
    bootstrap_n_iterations: int = 100
    bootstrap_confidence_level: float = 0.95
    bootstrap_log_frequency: int = 1  # Log bootstrap CI every N epochs

    # Random seed for reproducibility
    random_seed: int = 42

    # Class weight mode: 'balanced', 'sqrt_balanced', 'manual', or None
    class_weight_mode: Optional[str] = None
    # Manual class weights (used only if class_weight_mode='manual')
    manual_class_weights: Optional[Dict[int, float]] = None

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'


@dataclass
class ModelConfig:
    """Model architecture configuration for PANN classifier"""

    # PANN CNN14 configuration
    pann_model_name: str = "Cnn14"  # Options: Cnn14, Cnn10, Cnn6
    pann_sample_rate: int = 32000
    pann_window_size: int = 1024
    pann_hop_size: int = 320
    pann_mel_bins: int = 64
    pann_fmin: int = 50
    pann_fmax: int = 14000
    embedding_dim: int = 2048  # CNN14 output dimension (vs YAMNet's 1024)

    # Classifier head architecture
    hidden_units: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"  # Activation function
    use_batch_norm: bool = True
    dropout_rates: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.1])

    # Pretrained weights
    pretrained_weights_url: str = (
        "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    )
    pretrained_weights_path: Optional[str] = None  # If already downloaded

    def __post_init__(self):
        # Ensure dropout rates match hidden units
        if len(self.dropout_rates) != len(self.hidden_units):
            # Pad or truncate dropout rates
            if len(self.dropout_rates) < len(self.hidden_units):
                self.dropout_rates = self.dropout_rates + [0.1] * (
                    len(self.hidden_units) - len(self.dropout_rates)
                )
            else:
                self.dropout_rates = self.dropout_rates[: len(self.hidden_units)]
