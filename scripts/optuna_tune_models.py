"""
Lightweight Optuna hyperparameter tuning for COI separation models.

This script tunes the most important hyperparameters for each model (sudormrf, tuss, clapsep)
and saves the best configurations for fair model comparison.

REDESIGNED ARCHITECTURE:
-----------------------
This script NOW uses dedicated tuning configurations (configs/tuning/) instead of
modifying production training configs. This provides:

1. Clear separation: Tuning trials don't interfere with normal training runs
2. Explicit hyperparameter ranges: Documented in tuning config files
3. Fast iteration: Default 5 epochs per trial
4. Storage efficiency: Optional --no-save-checkpoints flag

STORAGE-EFFICIENT DESIGN (--no-save-checkpoints):
-------------------------------------------------
To save HPC storage during tuning, use the --no-save-checkpoints flag.
This will:
- Track hyperparameters and metrics in Optuna database (only ~few MB)
- Skip saving model checkpoints during trials (saves ~100+ checkpoint files)
- Save best hyperparameters to configs/tuned/ for later retraining

Recommended workflow:
1. Run tuning with --no-save-checkpoints to find optimal hyperparameters
2. Update training_config.yaml files with best parameters
3. Run full training to get final model weights with checkpoints

IMPORTANT NOTES ON MODEL TRAINING INTERFACES:
---------------------------------------------
Each model has a different command-line interface for training:

1. SuDoRMRF (train.py):
   - Accepts: --config <path_to_yaml>
   - Optionally: --resume <checkpoint_dir>
   - Solution: Create temporary YAML file with trial config

2. TUSS (train.py):
   - Reads config from hardcoded "training_config.yaml" in models/tuss/
   - Accepts: --device or --gpu to override device
   - Solution: Temporarily replace training_config.yaml, restore after trial
   - Note: Loads pretrained TUSS model from pretrained_path in config

3. CLAPSep (train_text_coi.py):
   - Accepts: --config <path_to_yaml>
   - Uses text prompts + LoRA fine-tuning for parameter-efficient adaptation
   - Solution: Create temporary YAML file with trial config (like SuDoRMRF)
   - Note: Requires --clap-checkpoint to load pretrained CLAP encoder

All models properly load their pretrained checkpoints because:
- Tuning configs specify pretrained_path and clap_checkpoint
- Deep copy preserves these paths when creating trial configs
- Trial configs inherit pretrained paths from tuning configs

Optimized for birds dataset with abundant samples:
- Uses only 5 epochs by default (converges quickly)
- Disables augmentations (augment_multiplier=1)
- Reduced warmup steps compared to full training
- Each trial takes ~2-5 minutes

Usage:
    # Tune all models WITHOUT saving checkpoints (recommended for HPC)
    python scripts/optuna_tune_models.py --n-trials 20 --target bird --no-save-checkpoints

    # Tune all models on birds dataset (5 trials, 5 epochs each)
    python scripts/optuna_tune_models.py --n-trials 5 --target bird

    # Tune specific model with more trials
    python scripts/optuna_tune_models.py --model sudormrf --n-trials 30 --target bird

    # Override epochs if needed (e.g., for smaller datasets)
    python scripts/optuna_tune_models.py --n-trials 20 --max-epochs 10 --target airplane
"""

import argparse
import io
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml

# Fix UTF-8 encoding for Windows Server compatibility
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", line_buffering=True
    )
if sys.stderr is not None and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", line_buffering=True
    )

# Ensure we can import from src
_script_dir = Path(__file__).parent
_src_dir = _script_dir.parent / "src"
_configs_dir = _script_dir.parent / "configs"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


# =============================================================================
# Hyperparameter space definitions
# =============================================================================


def sample_from_space(trial: optuna.Trial, param_name: str, space_config: Dict[str, Any]) -> Any:
    """
    Sample a hyperparameter value from its space definition.
    
    Args:
        trial: Optuna trial object
        param_name: Name of the parameter
        space_config: Configuration dict with 'type' and type-specific keys
        
    Returns:
        Sampled value from the hyperparameter space
    """
    param_type = space_config["type"]
    
    if param_type == "float":
        return trial.suggest_float(
            param_name,
            space_config["low"],
            space_config["high"],
            log=space_config.get("log", False)
        )
    elif param_type == "int":
        return trial.suggest_int(
            param_name,
            space_config["low"],
            space_config["high"],
            log=space_config.get("log", False)
        )
    elif param_type == "categorical":
        return trial.suggest_categorical(param_name, space_config["choices"])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def create_trial_config(
    trial: optuna.Trial,
    tuning_config: Dict[str, Any],
    max_epochs: int = None,
    save_checkpoints: bool = True
) -> Dict[str, Any]:
    """
    Create a trial configuration by sampling from hyperparameter spaces.
    
    Args:
        trial: Optuna trial object
        tuning_config: Tuning configuration loaded from configs/tuning/
        max_epochs: Override for number of epochs (None = use tuning config)
        save_checkpoints: Whether to save model checkpoints
        
    Returns:
        Complete configuration dict for this trial
    """
    import copy
    
    # Start with the tuning config (deep copy to avoid modifying original)
    config = copy.deepcopy(tuning_config)
    
    # Override epochs if specified
    if max_epochs is not None:
        config["tuning"]["num_epochs"] = max_epochs
    
    # Set checkpoint saving
    config["tuning"]["save_checkpoints"] = save_checkpoints
    
    # Sample hyperparameters from their defined spaces
    hyperparameter_space = config.get("hyperparameter_space", {})
    
    for param_name, space_config in hyperparameter_space.items():
        # Check if this parameter is conditional
        conditional_on = space_config.get("conditional_on")
        
        if conditional_on:
            # Only sample if condition is met
            condition_param = conditional_on["param"]
            condition_value = conditional_on["value"]
            
            # Get the value of the conditioning parameter (already sampled)
            # We need to check what was sampled for the condition param
            if condition_param in trial.params:
                actual_value = trial.params[condition_param]
                if actual_value == condition_value:
                    sampled_value = sample_from_space(trial, param_name, space_config)
                else:
                    # Use default from config
                    continue
            else:
                # Condition param not yet sampled, skip for now
                continue
        else:
            # Unconditional parameter, always sample
            sampled_value = sample_from_space(trial, param_name, space_config)
        
        # Update config with sampled value
        # Need to find where this parameter lives in the config structure
        # Convention: hyperparameters are in 'training' or 'model' sections
        if param_name in config.get("training", {}):
            config["training"][param_name] = sampled_value
        elif param_name in config.get("model", {}):
            config["model"][param_name] = sampled_value
        else:
            # Try to infer from the parameter name
            # Architecture params usually go in 'model', training params in 'training'
            arch_params = ["out_channels", "in_channels", "num_blocks", "enc_num_basis",
                          "num_head_conv_blocks", "embed_dim", "encoder_embed_dim",
                          "n_masker_layer", "d_attn", "freeze_backbone", "use_lora", "lora_rank"]
            
            if param_name in arch_params:
                config["model"][param_name] = sampled_value
            else:
                config["training"][param_name] = sampled_value
    
    # Copy tuning settings to training section for compatibility with training scripts
    config["training"]["num_epochs"] = config["tuning"]["num_epochs"]
    config["training"]["checkpoint_dir"] = config["tuning"]["checkpoint_dir"]
    config["training"]["validate_every_n_epochs"] = config["tuning"]["validate_every_n_epochs"]
    
    # Handle checkpoint saving for different models
    if not save_checkpoints:
        if "save_checkpoints" in config["training"]:
            config["training"]["save_checkpoints"] = False
        # For models that don't have save_checkpoints, use temp dir
        if "checkpoint_dir" in config["training"]:
            config["training"]["checkpoint_dir"] = tempfile.gettempdir() + "/optuna_temp_ckpt"
    
    return config


# =============================================================================
# Training and evaluation
# =============================================================================


def _convert_paths_to_strings(obj):
    """Recursively convert Path objects to strings in nested dicts/lists."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    else:
        return obj


def run_training(
    model_name: str,
    config: Dict[str, Any],
    trial_number: int,
    device: str = "cuda",
    timeout: int = 7200,
    save_checkpoints: bool = True,
) -> float:
    """
    Run training for a model with given config and return best validation metric.

    IMPORTANT: Each model has different command-line interfaces:
    - SuDoRMRF: Accepts --config argument for YAML config file
    - TUSS: Reads from hardcoded training_config.yaml (no --config arg)
    - CLAPSep: Accepts --config argument (uses train_text_coi.py)

    This function handles these differences by:
    1. SuDoRMRF: Writing config to temp file and passing via --config
    2. TUSS: Temporarily replacing training_config.yaml with trial config
    3. CLAPSep: Writing config to temp file and passing via --config

    All models properly load pretrained checkpoints because the tuning_config
    (which includes pretrained paths) is deep-copied before modification.

    Args:
        model_name: One of 'sudormrf', 'tuss', 'clapsep'
        config: Configuration dictionary (must include pretrained checkpoint paths)
        trial_number: Optuna trial number
        device: Device to use (e.g., 'cuda', 'cuda:0', 'cpu')
        timeout: Maximum training time in seconds
        save_checkpoints: Whether to save checkpoints

    Returns:
        Best validation metric (SI-SNR or SNR in dB)
    """
    try:
        # Convert all Path objects to strings to avoid YAML serialization issues
        config = _convert_paths_to_strings(config)
        
        # Extract key training settings for display
        num_epochs = config["training"].get("num_epochs", "UNKNOWN")
        lr = config["training"].get("lr", "UNKNOWN")
        
        print(f"\n{'=' * 80}")
        print(f"Trial {trial_number}: {model_name.upper()} on {device}")
        print(f"Epochs: {num_epochs} | LR: {lr:.2e}")
        if model_name == "sudormrf":
            num_blocks = config["model"].get("num_blocks", "N/A")
            num_head = config["model"].get("num_head_conv_blocks", "N/A")
            print(f"Architecture: num_blocks={num_blocks}, num_head_conv_blocks={num_head}")
        print(f"{'=' * 80}\n")
        sys.stdout.flush()
        
        # Determine training script path and build command
        if model_name == "sudormrf":
            # SuDoRMRF accepts --config argument
            # Device must be set in config, not command line
            config["training"]["device"] = device

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f, default_flow_style=False)
                config_path = f.name

            script = str(_src_dir / "models" / "sudormrf" / "train.py")
            cmd = [sys.executable, script, "--config", config_path]
            cleanup_path = config_path

        elif model_name == "tuss":
            # TUSS reads from hardcoded training_config.yaml - temporarily replace it
            # Also ensure device is set in config
            config["training"]["device"] = device

            tuss_config_path = _src_dir / "models" / "tuss" / "training_config.yaml"

            # Backup original config
            backup_path = tuss_config_path.with_suffix(".yaml.backup")
            if tuss_config_path.exists():
                import shutil
                shutil.copy2(tuss_config_path, backup_path)

            # Write trial config
            with open(tuss_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            script = str(_src_dir / "models" / "tuss" / "train.py")
            # TUSS can accept --device as CLI override, so use it for clarity
            cmd = [sys.executable, script, "--device", device]
            cleanup_path = None  # Will restore from backup instead

        elif model_name == "clapsep":
            # CLAPSep uses train_text_coi.py with text prompts and LoRA
            # Accepts --config argument (like SuDoRMRF)
            config["training"]["device"] = device

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f, default_flow_style=False)
                config_path = f.name

            script = str(_src_dir / "models" / "clapsep" / "train_text_coi.py")
            
            # Build command with config file and optional overrides
            cmd = [sys.executable, script, "--config", config_path, "--device", device]
            
            # Override specific settings via CLI if needed
            model_cfg = config.get("model", {})
            if model_cfg.get("use_lora", False):
                cmd.append("--use-lora")
                cmd.extend(["--lora-rank", str(model_cfg.get("lora_rank", 8))])
            else:
                cmd.append("--no-lora")
            
            if not model_cfg.get("freeze_encoder", False):
                cmd.append("--no-freeze-encoder")
            else:
                cmd.append("--freeze-encoder")
            
            cleanup_path = config_path

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Run training script and stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # Line buffered
        )

        # Collect output while streaming it to console
        output_lines = []
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                output_lines.append(line)
            
            # Wait for process to complete
            process.wait(timeout=timeout)
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise

        # Join all output for metric parsing
        full_output = "".join(output_lines)

        # Parse best validation metric from output
        metric = parse_best_metric(full_output, model_name)

        print(f"\nTrial {trial_number} completed with metric: {metric:.2f} dB\n")
        sys.stdout.flush()
        return metric

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_number} timed out after {timeout}s")
        return -100.0
    except Exception as e:
        print(f"Error in trial {trial_number}: {e}")
        import traceback
        traceback.print_exc()
        return -100.0
    finally:
        # Clean up temp files and restore backups
        if cleanup_path and model_name in ["sudormrf", "clapsep"]:
            # SuDoRMRF and CLAPSep use temp config files
            Path(cleanup_path).unlink(missing_ok=True)
        elif model_name == "tuss":
            # Restore original config
            tuss_config_path = _src_dir / "models" / "tuss" / "training_config.yaml"
            backup_path = tuss_config_path.with_suffix(".yaml.backup")
            if backup_path.exists():
                import shutil
                shutil.move(str(backup_path), str(tuss_config_path))


def parse_best_metric(output: str, model_name: str) -> float:
    """
    Parse the best validation metric from training output.

    Looks for patterns like:
    - "Best Val SI-SNR: 12.34 dB"
    - "Best val_sisnr: 12.34"
    - "best_val_snr=12.34"
    """
    lines = output.split("\n")

    # Search patterns for different models
    patterns = [
        "Best Val SI-SNR:",
        "Best val SI-SNR:",
        "Best Val SNR:",
        "Best val SNR:",
        "best_val_sisnr",
        "best_val_snr",
    ]

    for line in reversed(lines):  # Start from end (most recent)
        for pattern in patterns:
            if pattern.lower() in line.lower():
                try:
                    # Extract numeric value
                    parts = (
                        line.split(":")[-1].replace("dB", "").replace("=", " ").strip()
                    )
                    metric = float(parts.split()[0])
                    return metric
                except (ValueError, IndexError):
                    continue

    # If we can't find the metric, check for a JSON summary
    for line in reversed(lines):
        if line.strip().startswith("{") and "best_val" in line:
            try:
                data = json.loads(line)
                if "best_val_sisnr" in data:
                    return float(data["best_val_sisnr"])
                if "best_val_snr" in data:
                    return float(data["best_val_snr"])
            except:
                pass

    print(f"Warning: Could not parse validation metric from training output")
    return -100.0


# =============================================================================
# Optuna objectives
# =============================================================================


def create_objective(
    model_name: str,
    tuning_config: Dict[str, Any],
    max_epochs: int = None,
    device: str = "cuda",
    save_checkpoints: bool = True,
):
    """Create Optuna objective function for a model."""

    def objective(trial: optuna.Trial) -> float:
        # Create config with trial hyperparameters
        config = create_trial_config(
            trial,
            tuning_config,
            max_epochs=max_epochs,
            save_checkpoints=save_checkpoints
        )

        # Run training and get metric
        metric = run_training(
            model_name, config, trial.number, device=device, save_checkpoints=save_checkpoints
        )
        return metric

    return objective


# =============================================================================
# Main tuning function
# =============================================================================


def tune_model(
    model_name: str,
    target_class: str = "bird",
    n_trials: int = 20,
    max_epochs: int = None,
    device: str = "cuda",
    storage: str = None,
    study_name: str = None,
    csv_path: str = "src/models/tuss/checkpoints/.csv",
    save_checkpoints: bool = True,
) -> Dict[str, Any]:
    """
    Tune hyperparameters for a specific model.

    Args:
        model_name: One of 'sudormrf', 'tuss', 'clapsep'
        target_class: Target class for separation (e.g., 'bird', 'airplane')
        n_trials: Number of Optuna trials to run
        max_epochs: Maximum epochs per trial (None = use tuning config default)
        device: Device to use (e.g., 'cuda', 'cuda:0', 'cpu')
        storage: Optuna storage URL
        study_name: Name for the Optuna study
        csv_path: Path to dataset CSV file
        save_checkpoints: Whether to save model checkpoints during trials

    Returns:
        Best hyperparameters dictionary
    """
    # Load tuning config
    tuning_config_path = _configs_dir / "tuning" / f"{model_name}_tuning.yaml"
    if not tuning_config_path.exists():
        raise FileNotFoundError(f"Tuning config not found: {tuning_config_path}")

    with open(tuning_config_path) as f:
        tuning_config = yaml.safe_load(f)

    # Update target classes for the specified class
    if model_name == "tuss":
        tuning_config["data"]["target_classes"] = [
            [target_class, target_class.capitalize()]
        ]
        tuning_config["model"]["coi_prompts"] = [target_class]
    else:
        tuning_config["data"]["target_classes"] = [
            target_class,
            target_class.capitalize(),
        ]

    # Set dataset CSV path
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset CSV file does not exist: {csv_path}")
    print(f"Using dataset CSV: {csv_path}")
    tuning_config["data"]["df_path"] = csv_path

    # Create study
    if study_name is None:
        study_name = f"{model_name}_{target_class}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if storage is None:
        storage = "sqlite:///optuna_coi_tuning.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize SI-SNR or SNR
        load_if_exists=True,
    )

    # Print study info
    epochs_per_trial = max_epochs if max_epochs else tuning_config['tuning']['num_epochs']
    print(f"\n{'=' * 80}")
    print(f"Starting Optuna hyperparameter tuning for {model_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Target class: {target_class}")
    print(f"Number of trials: {n_trials}")
    print(f"⚠️  EPOCHS PER TRIAL: {epochs_per_trial}  ⚠️")
    print(f"Device: {device}")
    print(f"Save checkpoints: {save_checkpoints}")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage}")
    print(f"Tuning config: {tuning_config_path}")
    if epochs_per_trial != 5:
        print(f"\n⚠️  WARNING: Using {epochs_per_trial} epochs (default is 5) ⚠️")
    print(f"{'=' * 80}\n")

    # Run optimization
    objective = create_objective(
        model_name, tuning_config, max_epochs, device=device, save_checkpoints=save_checkpoints
    )
    study.optimize(objective, n_trials=n_trials)

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Optimization completed for {model_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (SI-SNR/SNR): {study.best_trial.value:.2f} dB")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 80}\n")

    # Create best config by applying best params to tuning config
    best_config = create_trial_config(study.best_trial, tuning_config, save_checkpoints=True)

    # Save best config
    output_dir = Path("configs/tuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{target_class}_best.yaml"

    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"Best config saved to: {output_path}")

    # Save study statistics
    stats_path = output_dir / f"{model_name}_{target_class}_tuning_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Optuna Study Statistics for {model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Study name: {study_name}\n")
        f.write(f"Target class: {target_class}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_trial.value:.2f} dB\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("Trial history:\n")
        for trial in study.trials:
            status = "✓" if trial.value == study.best_trial.value else " "
            f.write(f"{status} Trial {trial.number:3d}: {trial.value:7.2f} dB")
            if trial.value > -50:  # Only show params for non-failed trials
                f.write(f"  (lr={trial.params.get('lr', 'N/A'):.2e})")
            f.write("\n")

    print(f"Study statistics saved to: {stats_path}\n")

    return study.best_trial.params


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for COI separation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 trials (uses default 5 epochs from tuning config)
  python scripts/optuna_tune_models.py --n-trials 5 --target bird

  # Tune specific model with more trials
  python scripts/optuna_tune_models.py --model sudormrf --n-trials 30 --target bird

  # Use specific GPU
  python scripts/optuna_tune_models.py --n-trials 20 --target bird --gpu 1

  # Override epochs if needed (e.g., for smaller datasets)
  python scripts/optuna_tune_models.py --n-trials 20 --max-epochs 10 --target airplane
  
  # Save storage with --no-save-checkpoints (recommended for HPC)
  python scripts/optuna_tune_models.py --n-trials 20 --target bird --no-save-checkpoints
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sudormrf", "tuss", "clapsep", "all"],
        default="all",
        help="Which model to tune (default: all)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="bird",
        help="Target class for separation (default: bird)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials per model (default: 20)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs per trial (default: None, uses tuning config value of 5)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_coi_tuning.db",
        help="Optuna storage URL (default: sqlite:///optuna_coi_tuning.db)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training: 'cuda', 'cuda:0', 'cuda:1', etc. (default: cuda)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index (shorthand for --device cuda:N)",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Disable checkpoint saving during trials to save HPC storage. "
             "Only hyperparameters will be tracked in Optuna DB. "
             "Retrain with best config afterward to get model weights.",
    )

    args = parser.parse_args()

    # Handle GPU shorthand
    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"

    # Determine which models to tune
    models_to_tune = (
        ["sudormrf", "tuss", "clapsep"] if args.model == "all" else [args.model]
    )

    # Tune models
    results = {}
    for model_name in models_to_tune:
        try:
            best_params = tune_model(
                model_name=model_name,
                target_class=args.target,
                n_trials=args.n_trials,
                max_epochs=args.max_epochs,
                device=args.device,
                storage=args.storage,
                save_checkpoints=not args.no_save_checkpoints,
            )
            results[model_name] = best_params
        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"ERROR: Failed to tune {model_name}")
            print(f"{'=' * 80}")
            print(f"{e}")
            import traceback
            traceback.print_exc()
            print(f"{'=' * 80}\n")
            results[model_name] = None

    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)
    for model_name, params in results.items():
        if params is not None:
            print(f"\n{model_name.upper()} - SUCCESSFUL:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n{model_name.upper()} - FAILED")
    print("\n" + "=" * 80)
    print(f"\nTuned configs saved to: configs/tuned/")
    print("You can now train the models with these optimized hyperparameters!")
    print("=" * 80)


if __name__ == "__main__":
    main()
