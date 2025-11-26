"""
Configuration dataclass for YAMNet finetuning.

This module defines the configuration parameters for training the airplane
detection classifier on top of YAMNet features.
"""

from dataclasses import dataclass, field


@dataclass
class FinetuneConfig:
    """Configuration for YAMNet finetuning.

    Attributes:
        path_yamnet: Path to the yamnet_planes directory containing model files.
        path_data_train: Path to directory containing training audio data.
        path_data_csv: Path to CSV file with audio paths and labels.
        path_save: Path to save trained models and metadata.
        classes: List of class names for classification.
        patch_hop_seconds: Time hop between patches in seconds.
        min_sample_seconds: Minimum audio duration in seconds.
        max_sample_seconds: Maximum audio duration in seconds.
        num_augmentations: Number of augmentations per class [class0, class1].
        num_hidden: List of hidden layer sizes for the classifier.
        num_classes: Number of output classes.
        optimizer_type: Type of optimizer ('SGD' or 'Adam').
        learning_rate: Initial learning rate.
        momentum: Momentum for SGD optimizer.
        decay: Learning rate decay for SGD optimizer.
        nesterov: Whether to use Nesterov momentum.
        loss_function: Loss function name.
        epochs: Maximum number of training epochs.
        validation_split: Fraction of data for validation.
        feature_extraction_method: Method for feature extraction (0 or 1).
    """

    path_yamnet: str = ""
    path_data_train: str = ""
    path_data_csv: str = ""
    path_save: str = ""

    classes: list = field(default_factory=lambda: ["not_plane", "plane"])

    patch_hop_seconds: float = 0.096
    min_sample_seconds: float = 1.0
    max_sample_seconds: float = 1000.0

    num_augmentations: list = field(default_factory=lambda: [0, 0])

    num_hidden: list = field(default_factory=lambda: [1024])
    num_classes: int = 2
    features_input_size: int = 1024

    optimizer_type: str = "SGD"
    learning_rate: float = 0.001
    momentum: float = 0.9
    decay: float = 1e-6
    nesterov: bool = True

    loss_function: str = "sparse_categorical_crossentropy"
    metrics: list = field(default_factory=lambda: ["accuracy"])

    epochs: int = 10000
    validation_split: float = 0.1

    feature_extraction_method: int = 1

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not self.path_yamnet:
            raise ValueError("path_yamnet must be provided")
        if not self.path_data_train:
            raise ValueError("path_data_train must be provided")
        if not self.path_save:
            raise ValueError("path_save must be provided")
        if self.optimizer_type not in ("SGD", "Adam"):
            raise ValueError(f"Invalid optimizer_type: {self.optimizer_type}")
        if self.feature_extraction_method not in (0, 1):
            raise ValueError(
                f"Invalid feature_extraction_method: {self.feature_extraction_method}"
            )
        if self.validation_split < 0 or self.validation_split >= 1:
            raise ValueError(
                f"validation_split must be in [0, 1): {self.validation_split}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive: {self.epochs}")
        if len(self.num_augmentations) != len(self.classes):
            raise ValueError(
                "num_augmentations must have same length as classes"
            )
