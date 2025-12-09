"""Training pipeline and utilities"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import math
import numpy as np

from model import PlaneClassifier, load_yamnet, ModelConfig
from lazyloader import prepare_dataset
from config import TrainingConfig
from helpers import validate_dataset_files


def create_callbacks(phase: str, config: TrainingConfig) -> list:
    """Create training callbacks for a specific phase

    Args:
        phase: 'phase1' or 'phase2'
        config: TrainingConfig instance

    Returns:
        List of Keras callbacks
    """
    is_phase1 = phase == "phase1"
    patience = config.phase1_patience if is_phase1 else config.phase2_patience
    reduce_lr_patience = (
        config.phase1_reduce_lr_patience
        if is_phase1
        else config.phase2_reduce_lr_patience
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config.log_dir) / phase
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / f"best_model_{phase}.weights.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-8,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        keras.callbacks.CSVLogger(str(checkpoint_dir / f"training_{phase}.csv")),
    ]

    return callbacks


def compile_model(model: PlaneClassifier, learning_rate: float, config: TrainingConfig):
    """Compile model with optimizer, loss, and metrics

    Args:
        model: PlaneClassifier instance
        learning_rate: Learning rate for optimizer
        config: TrainingConfig with optimizer parameters
    """
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=config.weight_decay,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
            clipnorm=config.clipnorm,
        ),
        loss=keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=config.label_smoothing
        ),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
            keras.metrics.AUC(name="auc", from_logits=True),
            keras.metrics.Precision(name="precision", thresholds=0.0),
            keras.metrics.Recall(name="recall", thresholds=0.0),
        ],
    )


def train_plane_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[TrainingConfig] = None,
) -> Tuple[PlaneClassifier, Dict[str, Any], Dict[str, Any], list]:
    """Complete two-phase training pipeline with early stopping

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        config: TrainingConfig instance (uses defaults if None)

    Returns:
        Tuple of (trained_model, phase1_history, phase2_history, test_results)
    """
    if config is None:
        config = TrainingConfig()

    # Validate audio files first to identify corrupted files
    print("=" * 70)
    print("VALIDATING AUDIO FILES")
    print("=" * 70)

    print("\nValidating training files...")
    train_df = validate_dataset_files(train_df, config.filename_column, verbose=True)

    print("\nValidating validation files...")
    val_df = validate_dataset_files(val_df, config.filename_column, verbose=True)

    print("\nValidating test files...")
    test_df = validate_dataset_files(test_df, config.filename_column, verbose=True)

    # Prepare datasets
    print("\n" + "=" * 70)
    print("PREPARING DATASETS")
    print("=" * 70)

    train_dataset = prepare_dataset(
        train_df, config, shuffle=True, augment=True, repeat=True
    )
    val_dataset = prepare_dataset(
        val_df, config, shuffle=False, augment=False, repeat=False
    )
    test_dataset = prepare_dataset(
        test_df, config, shuffle=False, augment=False, repeat=False
    )

    # Calculate steps per epoch
    # When augmentation is enabled, dataset size doubles (original + augmented)
    effective_train_size = (
        len(train_df) * 2 if config.use_augmentation else len(train_df)
    )
    steps_per_epoch = math.ceil(effective_train_size / config.batch_size)
    validation_steps = math.ceil(len(val_df) / config.batch_size)

    # Calculate class weights for imbalanced data
    label_counts = train_df[config.label_column].value_counts()
    total = len(train_df)
    class_weight = {
        0: total / (2 * label_counts.get(0, total)),
        1: total / (2 * label_counts.get(1, total)),
    }

    print(f"Train samples: {len(train_df)}")
    if config.use_augmentation:
        print(
            f"  With augmentation: {effective_train_size} samples (original + augmented)"
        )
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Class distribution: {label_counts.to_dict()}")
    print(f"Class weights: {class_weight}")

    # Load YAMNet
    print("\nLoading YAMNet model...")
    yamnet = load_yamnet()

    # Initialize model
    model_config = ModelConfig(
        hidden_units=config.hidden_units,
        dropout_rates=[
            config.dropout_rate_1,
            config.dropout_rate_2,
            config.dropout_rate_3,
        ],
        l2_reg=config.l2_reg,
    )
    model = PlaneClassifier(yamnet, model_config, fine_tune=False)

    # =========================================================================
    # PHASE 1: Train classifier head with frozen YAMNet
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING CLASSIFIER HEAD (YAMNet frozen)")
    print("=" * 70)

    compile_model(model, config.phase1_lr, config)
    callbacks_phase1 = create_callbacks("phase1", config)

    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.phase1_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weight,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    print(f"\nPhase 1 completed after {len(history_phase1.history['loss'])} epochs")
    print(f"Best validation AUC: {max(history_phase1.history['val_auc']):.4f}")

    # =========================================================================
    # PHASE 2: Fine-tune entire model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNING ENTIRE MODEL")
    print("=" * 70)

    # Enable fine-tuning flag
    # Note: TensorFlow Hub models loaded with hub.load() don't have a trainable attribute,
    # but their variables will be included in model.trainable_variables automatically.
    # The lower learning rate in phase 2 allows gradual fine-tuning of all parameters.
    model.fine_tune = True

    print(f"Total trainable variables in model: {len(model.trainable_variables)}")

    # Recompile and train phase 2
    compile_model(model, config.phase2_lr, config)
    callbacks_phase2 = create_callbacks("phase2", config)

    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.phase2_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weight,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    print(f"\nPhase 2 completed after {len(history_phase2.history['loss'])} epochs")
    print(f"Best validation AUC: {max(history_phase2.history['val_auc']):.4f}")

    # =========================================================================
    # EVALUATE ON TEST SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    test_steps = math.ceil(len(test_df) / config.batch_size)
    test_results = model.evaluate(test_dataset, steps=test_steps, verbose=1)

    print("\nTest Results:")
    # Handle both list and dict return types from evaluate
    if isinstance(test_results, dict):
        # If evaluate returns a dict, use it directly
        for metric_name, value in test_results.items():
            print(f"  {metric_name}: {value:.4f}")

        # Calculate F1 score if precision and recall are available
        if "precision" in test_results and "recall" in test_results:
            precision = test_results["precision"]
            recall = test_results["recall"]
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
            print(f"  F1 Score: {f1_score:.4f}")
    else:
        # If evaluate returns a list, zip with metric names
        for metric_name, value in zip(model.metrics_names, test_results):
            print(f"  {metric_name}: {value:.4f}")

        # Try to calculate F1 score if precision and recall are in metrics
        try:
            precision_idx = model.metrics_names.index("precision")
            recall_idx = model.metrics_names.index("recall")
            precision = test_results[precision_idx]
            recall = test_results[recall_idx]
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
            print(f"  F1 Score: {f1_score:.4f}")
        except (ValueError, IndexError):
            print("  F1 Score: Unable to calculate (precision/recall not available)")

    # Save final model
    print("\n" + "=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)

    # Save weights instead of full model due to TensorFlow Hub serialization issues
    final_weights_path = Path(config.checkpoint_dir) / "final_model.weights.h5"
    model.save_weights(final_weights_path)
    print(f"Model weights saved to: {final_weights_path}")
    print("\nNote: To load the model, you'll need to:")
    print("  1. Load YAMNet: yamnet = load_yamnet()")
    print("  2. Create model: model = PlaneClassifier(yamnet, config)")
    print(f"  3. Load weights: model.load_weights('{final_weights_path}')")

    return model, history_phase1.history, history_phase2.history, test_results
