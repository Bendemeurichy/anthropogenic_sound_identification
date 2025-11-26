"""
YAMNet classifier model for airplane detection.

This module provides the classifier architecture that operates on top of
YAMNet feature embeddings.
"""

from tensorflow.keras import Model, layers


def create_classifier(
    input_size: int = 1024,
    num_hidden: list | None = None,
    num_classes: int = 2,
) -> Model:
    """Create a classifier model for YAMNet features.

    The classifier takes YAMNet feature embeddings (1024-dimensional vectors)
    and outputs class probabilities.

    Args:
        input_size: Size of input feature vector (default: 1024).
        num_hidden: List of hidden layer sizes. Defaults to [1024].
        num_classes: Number of output classes (default: 2).

    Returns:
        A compiled Keras Model.
    """
    if num_hidden is None:
        num_hidden = [1024]

    input_layer = layers.Input(shape=(input_size,))

    dense_layer = layers.Dense(num_hidden[0], activation=None)(input_layer)

    for idx_layer in range(1, len(num_hidden)):
        dense_layer = layers.Dense(num_hidden[idx_layer], activation=None)(
            dense_layer
        )

    classifier_layer = layers.Dense(num_classes, activation="softmax")(
        dense_layer
    )

    model = Model(inputs=input_layer, outputs=classifier_layer)
    return model
