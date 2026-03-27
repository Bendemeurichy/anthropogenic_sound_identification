"""Utility to inspect TUSS model checkpoints and extract prompt information."""

import argparse
import sys
from pathlib import Path

import torch
import yaml


def get_prompts_from_checkpoint(checkpoint_path: str | Path) -> dict:
    """Extract prompt information from a TUSS checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        
    Returns:
        Dictionary with:
            - 'coi_prompts': List of COI class prompts
            - 'bg_prompt': Background prompt name
            - 'all_prompts': All prompts found in the model state
            - 'prompt_shapes': Dict mapping prompt names to tensor shapes
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Try to get prompts from checkpoint metadata (saved in newer versions)
    coi_prompts = ckpt.get("coi_prompts", [])
    bg_prompt = ckpt.get("bg_prompt", "")
    all_prompts_meta = ckpt.get("all_prompts", [])
    
    # Extract prompts from model state dict
    model_state = ckpt.get("model_state_dict", {})
    prompts_in_state = {}
    
    for key, value in model_state.items():
        if key.startswith("separator.prompts."):
            prompt_name = key.replace("separator.prompts.", "")
            prompts_in_state[prompt_name] = value.shape
    
    return {
        "coi_prompts": coi_prompts,
        "bg_prompt": bg_prompt,
        "all_prompts_meta": all_prompts_meta,
        "prompts_in_state": prompts_in_state,
        "checkpoint_info": {
            "epoch": ckpt.get("epoch", "N/A"),
            "global_step": ckpt.get("global_step", "N/A"),
            "val_loss": ckpt.get("val_loss", "N/A"),
        },
    }


def print_checkpoint_info(checkpoint_path: str | Path, verbose: bool = False):
    """Print detailed information about a checkpoint's prompts."""
    info = get_prompts_from_checkpoint(checkpoint_path)
    
    print("\n" + "="*70)
    print("CHECKPOINT INFORMATION")
    print("="*70)
    
    print(f"\nTraining State:")
    print(f"  Epoch: {info['checkpoint_info']['epoch']}")
    print(f"  Global Step: {info['checkpoint_info']['global_step']}")
    print(f"  Validation Loss: {info['checkpoint_info']['val_loss']}")
    
    print(f"\nPrompts from Metadata:")
    if info['coi_prompts']:
        print(f"  COI Prompts ({len(info['coi_prompts'])}): {info['coi_prompts']}")
        print(f"  Background Prompt: {info['bg_prompt']}")
        print(f"  All Prompts: {info['all_prompts_meta']}")
    else:
        print("  (No metadata found - checkpoint may be from older version)")
    
    print(f"\nPrompts from Model State ({len(info['prompts_in_state'])} total):")
    for prompt_name in sorted(info['prompts_in_state'].keys()):
        shape = info['prompts_in_state'][prompt_name]
        if verbose:
            print(f"  - {prompt_name:20s} (shape: {shape})")
        else:
            print(f"  - {prompt_name}")
    
    # Identify COI vs background prompts
    coi_from_state = []
    bg_from_state = []
    
    if info['bg_prompt']:
        bg_name = info['bg_prompt']
        if bg_name in info['prompts_in_state']:
            bg_from_state.append(bg_name)
        coi_from_state = [p for p in info['prompts_in_state'].keys() if p != bg_name]
    elif info['coi_prompts']:
        coi_from_state = [p for p in info['coi_prompts'] if p in info['prompts_in_state']]
        bg_from_state = [p for p in info['prompts_in_state'].keys() if p not in coi_from_state]
    
    if coi_from_state or bg_from_state:
        print(f"\nClassification:")
        if coi_from_state:
            print(f"  COI Classes ({len(coi_from_state)}): {coi_from_state}")
        if bg_from_state:
            print(f"  Background: {bg_from_state}")
    
    print("\n" + "="*70)
    
    return info


def compare_with_config(checkpoint_path: str | Path, config_path: str | Path):
    """Compare checkpoint prompts with a training config file."""
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load checkpoint info
    ckpt_info = get_prompts_from_checkpoint(checkpoint_path)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config_coi_prompts = config.get("model", {}).get("coi_prompts", [])
    config_bg_prompt = config.get("model", {}).get("bg_prompt", "")
    config_all_prompts = config_coi_prompts + ([config_bg_prompt] if config_bg_prompt else [])
    
    print("\n" + "="*70)
    print("CONFIG vs CHECKPOINT COMPARISON")
    print("="*70)
    
    print(f"\nConfig file: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    print(f"\nPrompts in CONFIG: {config_all_prompts}")
    print(f"Prompts in CHECKPOINT: {sorted(ckpt_info['prompts_in_state'].keys())}")
    
    # Find differences
    config_set = set(config_all_prompts)
    ckpt_set = set(ckpt_info['prompts_in_state'].keys())
    
    already_trained = config_set & ckpt_set
    new_prompts = config_set - ckpt_set
    extra_in_ckpt = ckpt_set - config_set
    
    print(f"\nAnalysis:")
    if already_trained:
        print(f"  ✓ Already trained ({len(already_trained)}): {sorted(already_trained)}")
    if new_prompts:
        print(f"  + NEW in config ({len(new_prompts)}): {sorted(new_prompts)}")
    if extra_in_ckpt:
        print(f"  - In checkpoint but not in config ({len(extra_in_ckpt)}): {sorted(extra_in_ckpt)}")
    
    if not new_prompts:
        print(f"\n⚠ WARNING: All prompts in config already exist in checkpoint!")
        print(f"  You're about to retrain already-learned prompts.")
        print(f"  Consider adding NEW prompts to extend the model's capabilities.")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect TUSS model checkpoint and extract prompt information"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information including tensor shapes",
    )
    parser.add_argument(
        "-c", "--compare-config",
        type=str,
        metavar="CONFIG",
        help="Compare checkpoint with a training config file (training_config.yaml)",
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare_config:
            compare_with_config(args.checkpoint, args.compare_config)
        else:
            print_checkpoint_info(args.checkpoint, verbose=args.verbose)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
