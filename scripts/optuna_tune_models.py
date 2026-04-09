"""
Lightweight Optuna hyperparameter tuning for COI separation models.

This script tunes the most important hyperparameters for each model (sudormrf, tuss, clapsep)
and saves the best configurations for fair model comparison.

Optimized for birds dataset with abundant samples:
- Uses only 5 epochs by default (converges quickly)
- Disables augmentations (augment_multiplier=1)
- Reduced warmup steps (50-200 instead of 200-600)
- Each trial takes ~2-5 minutes

Usage:
    # Tune all models on birds dataset (5 trials, 5 epochs each)
    python scripts/optuna_tune_models.py --n-trials 5 --target bird

    # Tune specific model with more trials
    python scripts/optuna_tune_models.py --model sudormrf --n-trials 30 --target bird
    
    # Full tuning run (default 5 epochs)
    python scripts/optuna_tune_models.py --n-trials 20 --target bird
    
    # For smaller datasets, increase epochs
    python scripts/optuna_tune_models.py --n-trials 20 --max-epochs 50 --target airplane
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml

# Ensure we can import from src
_script_dir = Path(__file__).parent
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


# =============================================================================
# Hyperparameter space definitions
# =============================================================================

def create_sudormrf_trial_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create SuDoRMRF config with trial hyperparameters."""
    config = {k: v for k, v in base_config.items()}  # Deep copy
    
    # Most important architectural hyperparameters
    config["model"]["out_channels"] = trial.suggest_categorical("out_channels", [128, 256, 512])
    config["model"]["in_channels"] = trial.suggest_categorical("in_channels", [256, 512, 768])
    config["model"]["num_blocks"] = trial.suggest_int("num_blocks", 12, 20)
    config["model"]["enc_num_basis"] = trial.suggest_categorical("enc_num_basis", [512, 1024, 2048])
    config["model"]["num_head_conv_blocks"] = trial.suggest_int("num_head_conv_blocks", 1, 3)
    
    # Training hyperparameters
    config["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    config["training"]["class_weight"] = trial.suggest_float("class_weight", 1.0, 3.0)
    config["training"]["warmup_steps"] = trial.suggest_int("warmup_steps", 50, 200)
    config["training"]["grad_accum_steps"] = trial.suggest_categorical("grad_accum_steps", [8, 16, 32])
    
    # Disable augmentations for faster tuning with abundant bird data
    config["data"]["augment_multiplier"] = 1
    config["data"]["background_only_prob"] = 0.1
    
    return config


def create_tuss_trial_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create TUSS config with trial hyperparameters."""
    config = {k: v for k, v in base_config.items()}
    
    # TUSS-specific hyperparameters
    config["training"]["lr"] = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    config["training"]["coi_weight"] = trial.suggest_float("coi_weight", 1.0, 3.0)
    config["training"]["zero_ref_loss_weight"] = trial.suggest_float("zero_ref_loss_weight", 0.01, 0.5, log=True)
    config["training"]["warmup_steps"] = trial.suggest_int("warmup_steps", 50, 200)
    config["training"]["grad_accum_steps"] = trial.suggest_categorical("grad_accum_steps", [4, 8, 16])
    
    # Whether to freeze pretrained backbone
    config["model"]["freeze_backbone"] = trial.suggest_categorical("freeze_backbone", [True, False])
    
    # Disable augmentations for faster tuning with abundant bird data
    config["data"]["augment_multiplier"] = 1
    config["data"]["background_only_prob"] = 0.1
    
    return config


def create_clapsep_trial_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create CLAPSep config with trial hyperparameters."""
    config = {k: v for k, v in base_config.items()}
    
    # CLAPSep architecture hyperparameters  
    config["model"]["embed_dim"] = trial.suggest_categorical("embed_dim", [64, 128, 256])
    config["model"]["encoder_embed_dim"] = trial.suggest_categorical("encoder_embed_dim", [64, 128, 256])
    config["model"]["n_masker_layer"] = trial.suggest_int("n_masker_layer", 2, 5)
    config["model"]["d_attn"] = trial.suggest_categorical("d_attn", [320, 640, 1024])
    
    # Training hyperparameters
    config["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    config["training"]["class_weight"] = trial.suggest_float("class_weight", 1.0, 3.0)
    
    # Encoder fine-tuning strategy
    use_lora = trial.suggest_categorical("use_lora", [True, False])
    if use_lora:
        # LoRA fine-tuning (parameter-efficient)
        config["model"]["freeze_encoder"] = False
        config["model"]["use_lora"] = True
        config["model"]["lora_rank"] = trial.suggest_categorical("lora_rank", [4, 8, 16])
    else:
        # Freeze encoder completely (decoder-only training)
        config["model"]["freeze_encoder"] = True
        config["model"]["use_lora"] = False
    
    # Disable augmentations for faster tuning with abundant bird data
    config["data"]["augment_multiplier"] = 1
    config["data"]["background_only_prob"] = 0.1
    
    return config


# =============================================================================
# Training and evaluation
# =============================================================================

def run_training(
    model_name: str,
    config: Dict[str, Any],
    trial_number: int,
    device: str = "cuda",
    timeout: int = 7200,
) -> float:
    """
    Run training for a model with given config and return best validation metric.
    
    Args:
        model_name: One of 'sudormrf', 'tuss', 'clapsep'
        config: Configuration dictionary
        trial_number: Optuna trial number
        timeout: Maximum training time in seconds
        
    Returns:
        Best validation metric (SI-SNR or SNR in dB)
    """
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name
    
    try:
        # Determine training script path and build command
        if model_name == "sudormrf":
            script = str(_src_dir / "models" / "sudormrf" / "train.py")
            cmd = [sys.executable, script, "--config", config_path, "--device", device]
        elif model_name == "tuss":
            script = str(_src_dir / "models" / "tuss" / "train.py")
            cmd = [sys.executable, script, "--config", config_path, "--device", device]
        elif model_name == "clapsep":
            script = str(_src_dir / "models" / "clapsep" / "train_coi.py")
            cmd = [sys.executable, script, "--config", config_path, "--device", device]
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"\n{'='*80}")
        print(f"Trial {trial_number}: Running {model_name} on {device}")
        print(f"{'='*80}\n")
        
        # Run training script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        # Parse best validation metric from output
        metric = parse_best_metric(result.stdout, model_name)
        
        print(f"\nTrial {trial_number} completed with metric: {metric:.2f} dB\n")
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
        # Clean up temp file
        Path(config_path).unlink(missing_ok=True)


def parse_best_metric(output: str, model_name: str) -> float:
    """
    Parse the best validation metric from training output.
    
    Looks for patterns like:
    - "Best Val SI-SNR: 12.34 dB"
    - "Best val_sisnr: 12.34"
    - "best_val_snr=12.34"
    """
    lines = output.split('\n')
    
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
                    parts = line.split(':')[-1].replace('dB', '').replace('=', ' ').strip()
                    metric = float(parts.split()[0])
                    return metric
                except (ValueError, IndexError):
                    continue
    
    # If we can't find the metric, check for a JSON summary
    for line in reversed(lines):
        if line.strip().startswith('{') and 'best_val' in line:
            try:
                data = json.loads(line)
                if 'best_val_sisnr' in data:
                    return float(data['best_val_sisnr'])
                if 'best_val_snr' in data:
                    return float(data['best_val_snr'])
            except:
                pass
    
    print(f"Warning: Could not parse validation metric from training output")
    return -100.0


# =============================================================================
# Optuna objectives
# =============================================================================

def create_objective(model_name: str, base_config: Dict[str, Any], max_epochs: int = None, device: str = "cuda"):
    """Create Optuna objective function for a model."""
    
    def objective(trial: optuna.Trial) -> float:
        # Create config with trial hyperparameters
        if model_name == "sudormrf":
            config = create_sudormrf_trial_config(trial, base_config)
        elif model_name == "tuss":
            config = create_tuss_trial_config(trial, base_config)
        elif model_name == "clapsep":
            config = create_clapsep_trial_config(trial, base_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Override max epochs if specified (for quick testing)
        if max_epochs is not None:
            config["training"]["num_epochs"] = max_epochs
        
        # Run training and get metric
        metric = run_training(model_name, config, trial.number, device=device)
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
) -> Dict[str, Any]:
    """
    Tune hyperparameters for a specific model.
    
    Args:
        model_name: One of 'sudormrf', 'tuss', 'clapsep'
        target_class: Target class for separation (e.g., 'bird', 'airplane')
        n_trials: Number of Optuna trials to run
        max_epochs: Maximum epochs per trial (for quick testing)
        storage: Optuna storage URL
        study_name: Name for the Optuna study
        
    Returns:
        Best hyperparameters dictionary
    """
    # Load base config
    config_path = _src_dir / "models" / model_name / "training_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Update target classes for the specified class
    if model_name == "tuss":
        base_config["data"]["target_classes"] = [[target_class, target_class.capitalize()]]
        base_config["model"]["coi_prompts"] = [target_class]
    else:
        base_config["data"]["target_classes"] = [target_class, target_class.capitalize()]
    
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
    print(f"\n{'='*80}")
    print(f"Starting Optuna hyperparameter tuning for {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Target class: {target_class}")
    print(f"Number of trials: {n_trials}")
    print(f"Max epochs per trial: {max_epochs if max_epochs else 'from config'}")
    print(f"Device: {device}")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage}")
    print(f"{'='*80}\n")
    
    # Run optimization
    objective = create_objective(model_name, base_config, max_epochs, device=device)
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Optimization completed for {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (SI-SNR/SNR): {study.best_trial.value:.2f} dB")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Create best config
    if model_name == "sudormrf":
        best_config = create_sudormrf_trial_config(study.best_trial, base_config)
    elif model_name == "tuss":
        best_config = create_tuss_trial_config(study.best_trial, base_config)
    elif model_name == "clapsep":
        best_config = create_clapsep_trial_config(study.best_trial, base_config)
    
    # Save best config
    output_dir = Path("configs/tuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{target_class}_best.yaml"
    
    with open(output_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"Best config saved to: {output_path}")
    
    # Save study statistics
    stats_path = output_dir / f"{model_name}_{target_class}_tuning_stats.txt"
    with open(stats_path, 'w') as f:
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
  # Quick test with 5 trials (uses default 5 epochs)
  python scripts/optuna_tune_models.py --n-trials 5 --target bird
  
  # Tune specific model with more trials
  python scripts/optuna_tune_models.py --model sudormrf --n-trials 30 --target bird
  
  # Use specific GPU
  python scripts/optuna_tune_models.py --n-trials 20 --target bird --gpu 1
  # Or equivalently:
  python scripts/optuna_tune_models.py --n-trials 20 --target bird --device cuda:1
  
  # Override epochs if needed (e.g., for smaller datasets)
  python scripts/optuna_tune_models.py --n-trials 20 --max-epochs 50 --target airplane
        """
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
        default=5,
        help="Maximum epochs per trial (default: 5, optimized for abundant bird data)",
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
    
    args = parser.parse_args()
    
    # Handle GPU shorthand
    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"
    
    # Determine which models to tune
    models_to_tune = ["sudormrf", "tuss", "clapsep"] if args.model == "all" else [args.model]
    
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
            )
            results[model_name] = best_params
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Failed to tune {model_name}")
            print(f"{'='*80}")
            print(f"{e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
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
