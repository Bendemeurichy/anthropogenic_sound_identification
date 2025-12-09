"""Training pipeline and utilities"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import math
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from model import PlaneClassifier, load_yamnet, ModelConfig
from lazyloader import prepare_dataset
from config import TrainingConfig
from helpers import validate_dataset_files


class BootstrapPRAUCCallback(keras.callbacks.Callback):
    """Custom callback to compute bootstrap confidence intervals for validation PR-AUC"""

    def __init__(
        self,
        validation_data: tf.data.Dataset,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        log_frequency: int = 1,
    ):
        super().__init__()
        self.validation_data = validation_data
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.log_frequency = log_frequency
        self.bootstrap_results: list = []

    def on_epoch_end(self, epoch, logs=None):
        # Only compute bootstrap CI every N epochs
        if (epoch + 1) % self.log_frequency != 0:
            return

        # Collect all validation predictions and labels
        y_true_list = []
        y_pred_list = []

        for batch_x, batch_y in self.validation_data:
            y_pred = self.model.predict(batch_x, verbose=0)
            y_true_list.append(batch_y.numpy())
            y_pred_list.append(y_pred.flatten())

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)

        # Compute bootstrap confidence interval
        n_samples = len(y_true)
        bootstrap_scores = []

        np.random.seed(42 + epoch)  # Deterministic but different per epoch

        for _ in range(self.n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Compute PR-AUC for this bootstrap sample
            try:
                precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_boot)
                pr_auc = auc(recall, precision)
                bootstrap_scores.append(pr_auc)
            except Exception:
                continue

        if bootstrap_scores:
            bootstrap_scores = np.array(bootstrap_scores)
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_scores, lower_percentile)
            ci_upper = np.percentile(bootstrap_scores, upper_percentile)
            mean_score = np.mean(bootstrap_scores)

            # Store results
            self.bootstrap_results.append(
                {
                    "epoch": epoch + 1,
                    "mean": mean_score,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

            # Log to console
            print(
                f"\n  Bootstrap PR-AUC (n={self.n_iterations}): "
                f"{mean_score:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] "
                f"(CI width: {ci_upper - ci_lower:.4f})"
            )


def compute_optimal_threshold(
    model: PlaneClassifier,
    val_dataset: tf.data.Dataset,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """Compute optimal threshold on validation set based on PR curve

    Args:
        model: Trained model
        val_dataset: Validation dataset
        metric: 'f1' for max F1-score or 'precision_at_recall' for precision at specific recall

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    # Collect predictions and labels
    y_true_list = []
    y_pred_list = []

    for batch_x, batch_y in val_dataset:
        y_pred = model.predict(batch_x, verbose=0)
        y_true_list.append(batch_y.numpy())
        y_pred_list.append(y_pred.flatten())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    if metric == "f1":
        # Compute F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        )

        metrics = {
            "threshold": float(optimal_threshold),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1_score": float(f1_scores[best_idx]),
        }
    else:
        # Default to threshold at max F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        )
        metrics = {
            "threshold": float(optimal_threshold),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1_score": float(f1_scores[best_idx]),
        }

    return optimal_threshold, metrics


def create_callbacks(
    phase: str,
    config: TrainingConfig,
    validation_data: Optional[tf.data.Dataset] = None,
) -> Tuple[list, Optional[BootstrapPRAUCCallback]]:
    """Create training callbacks for a specific phase

    Args:
        phase: 'phase1' or 'phase2'
        config: TrainingConfig instance
        validation_data: Validation dataset for bootstrap CI computation

    Returns:
        Tuple of (callbacks_list, bootstrap_callback)
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
            monitor="val_pr_auc",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / f"best_model_{phase}.weights.h5"),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-8,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        keras.callbacks.CSVLogger(str(checkpoint_dir / f"training_{phase}.csv")),
    ]

    # Add bootstrap CI callback if enabled and validation data provided
    bootstrap_callback = None
    if config.bootstrap_enabled and validation_data is not None:
        bootstrap_callback = BootstrapPRAUCCallback(
            validation_data=validation_data,
            n_iterations=config.bootstrap_n_iterations,
            confidence_level=config.bootstrap_confidence_level,
            log_frequency=config.bootstrap_log_frequency,
        )
        callbacks.append(bootstrap_callback)

    return callbacks, bootstrap_callback


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
            keras.metrics.AUC(name="pr_auc", curve="PR", from_logits=True),
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

    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

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
    # Let Keras handle validation steps automatically (no need to specify)

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
    callbacks_phase1, bootstrap_cb1 = create_callbacks("phase1", config, val_dataset)

    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.phase1_epochs,
        steps_per_epoch=steps_per_epoch,
        # Let Keras determine validation_steps automatically
        class_weight=class_weight,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    print(f"\nPhase 1 completed after {len(history_phase1.history['loss'])} epochs")
    print(f"Best validation AUC: {max(history_phase1.history['val_auc']):.4f}")
    print(f"Best validation PR-AUC: {max(history_phase1.history['val_pr_auc']):.4f}")

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
    callbacks_phase2, bootstrap_cb2 = create_callbacks("phase2", config, val_dataset)

    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.phase2_epochs,
        steps_per_epoch=steps_per_epoch,
        # Let Keras determine validation_steps automatically
        class_weight=class_weight,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    print(f"\nPhase 2 completed after {len(history_phase2.history['loss'])} epochs")
    print(f"Best validation AUC: {max(history_phase2.history['val_auc']):.4f}")
    print(f"Best validation PR-AUC: {max(history_phase2.history['val_pr_auc']):.4f}")

    # =========================================================================
    # COMPUTE OPTIMAL THRESHOLD ON VALIDATION SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPUTING OPTIMAL THRESHOLD ON VALIDATION SET")
    print("=" * 70)

    optimal_threshold, threshold_metrics = compute_optimal_threshold(
        model, val_dataset, metric="f1"
    )

    print(f"\nOptimal threshold (max F1): {optimal_threshold:.4f}")
    print(f"  Precision: {threshold_metrics['precision']:.4f}")
    print(f"  Recall:    {threshold_metrics['recall']:.4f}")
    print(f"  F1 Score:  {threshold_metrics['f1_score']:.4f}")

    # =========================================================================
    # EVALUATE ON TEST SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    test_results = model.evaluate(test_dataset, verbose=1)

    print("\nTest Results (default threshold=0.0):")
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

    # Compute adjusted metrics with optimal threshold
    print(f"\nTest Results (adjusted threshold={optimal_threshold:.4f}):")
    y_true_test = []
    y_pred_test = []
    for batch_x, batch_y in test_dataset:
        y_pred = model.predict(batch_x, verbose=0)
        y_true_test.append(batch_y.numpy())
        y_pred_test.append(y_pred.flatten())

    y_true_test = np.concatenate(y_true_test)
    y_pred_test = np.concatenate(y_pred_test)
    y_pred_binary = (y_pred_test >= optimal_threshold).astype(int)

    # Compute adjusted precision, recall, and F1
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score as compute_f1_score,
    )

    adj_precision = precision_score(y_true_test, y_pred_binary)
    adj_recall = recall_score(y_true_test, y_pred_binary)
    adj_f1 = compute_f1_score(y_true_test, y_pred_binary)

    print(f"  Precision: {adj_precision:.4f}")
    print(f"  Recall:    {adj_recall:.4f}")
    print(f"  F1 Score:  {adj_f1:.4f}")

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
