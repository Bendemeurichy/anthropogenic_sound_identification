"""
Utility to load a trained PlaneClassifier model from saved weights.
"""

from pathlib import Path
from model import PlaneClassifier, load_yamnet, ModelConfig
from config import TrainingConfig


def load_trained_model(weights_path: str, config: TrainingConfig = None):
    """Load a trained PlaneClassifier model from weights file.

    Args:
        weights_path: Path to .weights.h5 file
        config: TrainingConfig with model architecture parameters
                (must match the architecture used during training)

    Returns:
        Loaded PlaneClassifier model

    Example:
        ```python
        from model_loader import load_trained_model
        from config import TrainingConfig

        config = TrainingConfig()
        model = load_trained_model("checkpoints/final_model.weights.h5", config)

        # Now you can use the model for inference
        predictions = model.predict(test_dataset)
        ```
    """
    if config is None:
        config = TrainingConfig()

    # Load YAMNet
    print("Loading YAMNet from TensorFlow Hub...")
    yamnet = load_yamnet()

    # Create model with same architecture
    model_config = ModelConfig(
        hidden_units=config.hidden_units,
        dropout_rates=[
            config.dropout_rate_1,
            config.dropout_rate_2,
            config.dropout_rate_3,
        ],
        l2_reg=config.l2_reg,
    )

    print("Creating PlaneClassifier model...")
    model = PlaneClassifier(yamnet, model_config, fine_tune=False)

    # Build the model by calling it once (required before loading weights)
    import tensorflow as tf
    import numpy as np

    # Create dummy input with correct shape (batch_size, samples)
    dummy_input = tf.constant(
        np.zeros((1, int(config.sample_rate * config.audio_duration)), dtype=np.float32)
    )
    _ = model(dummy_input, training=False)

    # Load weights
    print(f"Loading weights from {weights_path}...")
    model.load_weights(weights_path)

    print("Model loaded successfully!")
    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_loader.py <path_to_weights.h5>")
        print("\nExample:")
        print("  python model_loader.py checkpoints/final_model.weights.h5")
        sys.exit(1)

    weights_path = sys.argv[1]

    if not Path(weights_path).exists():
        print(f"Error: Weights file not found: {weights_path}")
        sys.exit(1)

    config = TrainingConfig()
    model = load_trained_model(weights_path, config)

    print("\nModel summary:")
    model.summary()
