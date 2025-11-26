"""
Main training script for YAMNet airplane detection finetuning.

This module provides the main training loop and utilities for finetuning
the YAMNet model for airplane sound detection.
"""

import datetime
import os
import sys
import random
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tqdm import tqdm

from .config import FinetuneConfig
from .model import create_classifier
from .data import (
    balance_classes,
    data_augmentation,
    load_features_from_csv,
    save_features,
)
from .callbacks import TrainingPlot


def setup_tensorflow() -> None:
    """Configure TensorFlow for training.

    Disables eager execution and sets GPU memory allocation.
    """
    tf.compat.v1.disable_eager_execution()

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
    tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def load_yamnet_features(path_yamnet: str, params):
    """Load the YAMNet feature extraction model.

    Args:
        path_yamnet: Path to yamnet_planes directory.
        params: YAMNet parameters module.

    Returns:
        Loaded YAMNet feature extraction model.
    """
    sys.path.insert(0, path_yamnet)

    import yamnet_modified

    yamnet_features = yamnet_modified.yamnet_frames_model(params)
    yamnet_features.load_weights(os.path.join(path_yamnet, "yamnet.h5"))

    return yamnet_features


def create_optimizer(config: FinetuneConfig):
    """Create optimizer based on configuration.

    Args:
        config: Training configuration.

    Returns:
        Configured optimizer instance.
    """
    if config.optimizer_type == "Adam":
        return Adam(learning_rate=config.learning_rate)
    elif config.optimizer_type == "SGD":
        return SGD(
            learning_rate=config.learning_rate,
            decay=config.decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


def extract_all_features(
    config: FinetuneConfig, yamnet_features, params
) -> None:
    """Extract and save features for all audio files.

    Args:
        config: Training configuration.
        yamnet_features: YAMNet feature extraction model.
        params: YAMNet parameters module.
    """
    patch_hop_seconds_str = str(config.patch_hop_seconds).replace(".", "")

    # Create features directory if needed
    features_dir = os.path.join(config.path_data_train, "features", "yamnet")
    os.makedirs(features_dir, exist_ok=True)

    df_data_csv = pd.read_csv(config.path_data_csv, header=None)
    classes = np.unique(df_data_csv.iloc[:, 1]).tolist()

    for class_idx, class_label in enumerate(classes):
        df_data_csv_class = df_data_csv[df_data_csv.iloc[:, 1] == class_label]
        path_audios = df_data_csv_class.iloc[:, 0].tolist()

        print(f"\nExtracting features for class: {class_label}")
        for path_audio in tqdm(path_audios):
            save_features(
                path_audio,
                params.SAMPLE_RATE,
                config.min_sample_seconds,
                config.path_data_train,
                patch_hop_seconds_str,
                config.num_augmentations,
                class_idx,
                yamnet_features,
            )


def load_training_data(
    config: FinetuneConfig, yamnet_features, params
) -> tuple[list, list]:
    """Load training data based on configuration.

    Args:
        config: Training configuration.
        yamnet_features: YAMNet feature extraction model.
        params: YAMNet parameters module.

    Returns:
        Tuple of (features, labels).
    """
    patch_hop_seconds_str = str(config.patch_hop_seconds).replace(".", "")

    if config.feature_extraction_method == 0:
        features, labels = data_augmentation(
            config.path_data_train,
            config.classes,
            yamnet_features,
            num_augmentations=config.num_augmentations,
            min_sample_seconds=config.min_sample_seconds,
            max_sample_seconds=config.max_sample_seconds,
            desired_sr=params.SAMPLE_RATE,
        )

        # Randomize and balance
        idxs = list(range(len(labels)))
        random.shuffle(idxs)
        features = [features[i] for i in idxs]
        labels = [labels[i] for i in idxs]

        features = np.array(features)
        labels = np.array(labels)

        # Balance classes
        _, counts = np.unique(labels, return_counts=True)
        idx_locs_delete = []

        for idx in np.unique(labels):
            idx_loc = np.where(labels == idx)[0]
            if len(idx_loc) > counts.min():
                idx_locs_delete = np.append(
                    idx_locs_delete, idx_loc[counts.min() - 1 : -1]
                )

        labels = np.delete(labels, idx_locs_delete.astype(int))
        features = np.delete(features, idx_locs_delete.astype(int), axis=0)

    elif config.feature_extraction_method == 1:
        features, labels = load_features_from_csv(
            config.path_data_csv,
            config.path_data_train,
            patch_hop_seconds_str,
            config.num_augmentations,
        )

        print("Balancing class features.\n")
        _, counts = np.unique(labels, return_counts=True)
        print(f"Sample distribution per class (before balancing): {counts}\n")

        features, labels = balance_classes(features, labels)

        _, counts = np.unique(labels, return_counts=True)
        print(f"Sample size: {len(features)}")
        print(f"Label size:  {len(labels)}")
        print(f"Sample distribution per class (after balancing): {counts}\n")

        features = [features]
        labels = [labels]

    return features, labels


def train(config: FinetuneConfig) -> dict:
    """Run the finetuning training loop.

    Args:
        config: Training configuration.

    Returns:
        Dictionary containing training history and metadata.
    """
    config.validate()

    setup_tensorflow()

    # Set up paths
    sys.path.insert(0, config.path_yamnet)
    import yamnet_original.params as params

    params.PATCH_HOP_SECONDS = config.patch_hop_seconds

    patch_hop_seconds_str = str(config.patch_hop_seconds).replace(".", "")

    # Load YAMNet feature extractor
    print("Loading YAMNet feature extraction model...")
    yamnet_features = load_yamnet_features(config.path_yamnet, params)

    # Extract features if needed
    if config.feature_extraction_method == 1 and config.path_data_csv:
        print("\nExtracting features from audio files...")
        extract_all_features(config, yamnet_features, params)

    # Load training data
    print("\nLoading training data...")
    features, labels = load_training_data(config, yamnet_features, params)

    _, counts = np.unique(labels if isinstance(labels, np.ndarray) else labels[0], return_counts=True)
    print(f"Sample distribution per class: {counts}\n")

    # Create classifier model
    print("Creating classifier model...")
    classifier = create_classifier(
        input_size=config.features_input_size,
        num_hidden=config.num_hidden,
        num_classes=config.num_classes,
    )

    # Create optimizer
    optimizer = create_optimizer(config)

    # Compile model
    classifier.compile(
        optimizer=optimizer, loss=config.loss_function, metrics=config.metrics
    )

    classifier.summary()

    # Set up paths for saving
    os.makedirs(config.path_save, exist_ok=True)
    time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    path_model = os.path.join(
        config.path_save,
        f"{time_now}_yamnet_{patch_hop_seconds_str}.hdf5",
    )
    path_plot = os.path.join(
        config.path_save,
        f"{time_now}_progress_{patch_hop_seconds_str}.png",
    )

    # Set up callbacks
    save_best = ModelCheckpoint(
        path_model, save_best_only=True, monitor="val_loss", mode="min"
    )
    plot_metrics = TrainingPlot(path_plot)

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    time_start = time()
    history = classifier.fit(
        features,
        labels,
        epochs=config.epochs,
        validation_split=config.validation_split,
        callbacks=[save_best, plot_metrics],
    )
    time_end = time()
    time_train = round(time_end - time_start)

    # Save metadata
    train_loss = history.history["loss"]
    train_acc = history.history.get("accuracy", history.history.get("acc", []))
    val_loss = history.history["val_loss"]
    val_acc = history.history.get("val_accuracy", history.history.get("val_acc", []))

    metadata_headers = [
        "path_data_train",
        "patch_hop_seconds",
        "features_count",
        "augmentations",
        "classifier_config",
        "optimizer",
        "optimizer_params",
        "loss_function",
        "path_model",
        "epochs",
        "val_split",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "training_time_seconds",
    ]

    opt_params = (
        [config.learning_rate]
        if config.optimizer_type == "Adam"
        else [config.learning_rate, config.decay, config.momentum, config.nesterov]
    )

    metadata_values = [
        config.path_data_train,
        config.patch_hop_seconds,
        counts.tolist(),
        config.num_augmentations,
        config.num_hidden,
        config.optimizer_type,
        opt_params,
        config.loss_function,
        path_model,
        config.epochs,
        config.validation_split,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        time_train,
    ]

    metadata_df = pd.DataFrame([metadata_values], columns=metadata_headers)
    path_metadata = os.path.join(
        config.path_save,
        f"{time_now}_metadata_{patch_hop_seconds_str}.csv",
    )
    metadata_df.to_csv(path_metadata, index=False)

    print("\n" + "=" * 50)
    print(
        f"Training complete: {config.epochs} epochs in "
        f"{round(time_train / 3600, 3)} hours "
        f"({time_train / config.epochs:.2f} seconds per epoch)"
    )
    print(f"Model saved to: {path_model}")
    print(f"Metadata saved to: {path_metadata}")
    print("=" * 50 + "\n")

    return {
        "history": history.history,
        "path_model": path_model,
        "path_metadata": path_metadata,
        "training_time": time_train,
    }
