"""Model architecture definition"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from typing import Optional
from config import ModelConfig


class PlaneClassifier(keras.Model):
    """Binary classifier for plane/no-plane audio using YAMNet backbone

    Args:
        yamnet_model: Pre-loaded YAMNet model from TensorFlow Hub
        config: ModelConfig instance with architecture parameters
        fine_tune: Whether YAMNet backbone is trainable
    """

    def __init__(
        self,
        yamnet_model,
        config: Optional[ModelConfig] = None,
        fine_tune: bool = False,
    ):
        super().__init__()

        if config is None:
            config = ModelConfig()

        self.config = config
        self.yamnet = yamnet_model

        # Build classification head
        layers = []
        # Type assertion after None check above guarantees these are not None
        assert config.hidden_units is not None
        assert config.dropout_rates is not None
        for i, (units, dropout) in enumerate(
            zip(config.hidden_units, config.dropout_rates)
        ):
            layers.append(
                keras.layers.Dense(
                    units,
                    activation=config.activation,
                    kernel_regularizer=keras.regularizers.l2(config.l2_reg),
                    name=f"dense_{i}",
                )
            )

            if config.use_batch_norm:
                layers.append(keras.layers.BatchNormalization(name=f"bn_{i}"))

            layers.append(keras.layers.Dropout(dropout, name=f"dropout_{i}"))

        # Output layer (logits)
        layers.append(keras.layers.Dense(1, name="output"))

        self.classifier = keras.Sequential(layers, name="classifier_head")

    def call(self, inputs, training=False):
        """Forward pass through the model

        Args:
            inputs: Audio waveform tensor with shape (batch_size, samples)
            training: Whether in training mode

        Returns:
            Logits for binary classification with shape (batch_size, 1)
        """

        # YAMNet expects 1D waveforms, so we need to process each sample in the batch
        def process_single_waveform(waveform):
            # YAMNet returns (scores, embeddings, spectrogram)
            _, embeddings, _ = self.yamnet(waveform)
            # Average pool across time dimension to get a single embedding vector
            pooled = tf.reduce_mean(embeddings, axis=0)
            return pooled

        # Apply YAMNet to each sample in the batch
        batch_embeddings = tf.map_fn(
            process_single_waveform,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(1024,), dtype=tf.float32),
        )

        # Classification head
        logits = self.classifier(batch_embeddings, training=training)

        return logits

    def get_config(self):
        """Return model configuration for serialization"""
        return {"config": self.config, "fine_tune": self.yamnet.trainable}


def load_yamnet(url: str = "https://tfhub.dev/google/yamnet/1"):
    """Load pre-trained YAMNet model from TensorFlow Hub

    Args:
        url: TensorFlow Hub URL for YAMNet

    Returns:
        YAMNet model
    """
    return hub.load(url)
