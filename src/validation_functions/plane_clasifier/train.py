"""Training pipeline and utilities"""

from tensorflow import keras
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from model import PlaneClassifier, load_yamnet, ModelConfig
from lazyloader import prepare_dataset
from config import TrainingConfig


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
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / f"best_model_{phase}.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-8,
            verbose=1,
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
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", from_logits=True),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
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

    # Prepare datasets
    print("=" * 70)
    print("PREPARING DATASETS")
    print("=" * 70)

    train_dataset = prepare_dataset(train_df, config, shuffle=True, augment=True)
    val_dataset = prepare_dataset(val_df, config, shuffle=False, augment=False)
    test_dataset = prepare_dataset(test_df, config, shuffle=False, augment=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

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

    # Unfreeze YAMNet
    model.yamnet.trainable = True

    # Recompile with lower learning rate
    compile_model(model, config.phase2_lr, config)
    callbacks_phase2 = create_callbacks("phase2", config)

    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.phase2_epochs,
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

    test_results = model.evaluate(test_dataset, verbose=1)

    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")

    # Calculate F1 score
    precision_idx = model.metrics_names.index("precision")
    recall_idx = model.metrics_names.index("recall")
    precision = test_results[precision_idx]
    recall = test_results[recall_idx]
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    print(f"  F1 Score: {f1_score:.4f}")

    # Save final model
    print("\n" + "=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)

    final_model_path = Path(config.checkpoint_dir) / "final_model.keras"
    model.save(final_model_path)
    print(f"Model saved to: {final_model_path}")

    return model, history_phase1.history, history_phase2.history, test_results
