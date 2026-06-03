"""Utility to inspect TUSS model checkpoints and extract prompt information."""

import argparse
import sys
from pathlib import Path

import numpy as np
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
            # Extract just the prompt name (use replace with count=1 to be explicit)
            prompt_name = key.replace("separator.prompts.", "", 1)
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


def _cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = (a_flat @ b_flat).item()
    norm_a = torch.norm(a_flat).item()
    norm_b = torch.norm(b_flat).item()
    return dot / (norm_a * norm_b + 1e-8)


def _euclidean_distance(a, b):
    return torch.norm(a.flatten() - b.flatten()).item()


def _load_and_compute_divergence(checkpoint_path, target_prompts=None):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model_state_dict", {})

    prompts = {}
    for key, value in model_state.items():
        if key.startswith("separator.prompts."):
            prompt_name = key.replace("separator.prompts.", "", 1)
            prompts[prompt_name] = value

    if not prompts:
        print("❌ No prompts found in checkpoint!")
        return None

    if target_prompts:
        prompts = {name: vec for name, vec in prompts.items() if name in target_prompts}
        if not prompts:
            print(f"❌ None of the target prompts {target_prompts} found in checkpoint!")
            return None

    prompt_names = sorted(prompts.keys())
    n = len(prompt_names)

    if n < 2:
        print("\n⚠ Need at least 2 prompts for divergence analysis")
        return None

    cosine_matrix = np.zeros((n, n))
    euclidean_matrix = np.zeros((n, n))
    similarities = []
    distances = []
    pairwise_info = []

    for i in range(n):
        for j in range(n):
            if i == j:
                cosine_matrix[i, j] = 1.0
                euclidean_matrix[i, j] = 0.0
            else:
                sim = _cosine_similarity(prompts[prompt_names[i]], prompts[prompt_names[j]])
                dist = _euclidean_distance(prompts[prompt_names[i]], prompts[prompt_names[j]])
                cosine_matrix[i, j] = sim
                euclidean_matrix[i, j] = dist
                if i < j:
                    similarities.append(sim)
                    distances.append(dist)
                    pairwise_info.append((prompt_names[i], prompt_names[j], sim, dist))

    pairwise_info.sort(key=lambda x: x[2], reverse=True)

    return {
        "prompts": prompts,
        "prompt_names": prompt_names,
        "n": n,
        "cosine_matrix": cosine_matrix,
        "euclidean_matrix": euclidean_matrix,
        "similarities": similarities,
        "distances": distances,
        "pairwise_info": pairwise_info,
        "avg_sim": float(np.mean(similarities)),
        "std_sim": float(np.std(similarities)),
        "min_sim": float(np.min(similarities)),
        "max_sim": float(np.max(similarities)),
        "avg_dist": float(np.mean(distances)),
        "std_dist": float(np.std(distances)),
        "min_dist": float(np.min(distances)),
        "max_dist": float(np.max(distances)),
        "n_pairs": len(similarities),
        "ckpt": ckpt,
    }


def _format_divergence_tables(results):
    prompt_names = results["prompt_names"]
    cosine_matrix = results["cosine_matrix"]
    euclidean_matrix = results["euclidean_matrix"]
    n = results["n"]

    lines = []

    lines.append(f"\n{'─'*70}")
    lines.append("Cosine Similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
    lines.append(f"{'─'*70}")

    header = f"{'':20}"
    for name in prompt_names:
        header += f"{name[:15]:>15}"
    lines.append(header)

    for i, name_i in enumerate(prompt_names):
        row = f"{name_i:20}"
        for j in range(n):
            if i == j:
                row += f"{'1.0000':>15}"
            else:
                sim = cosine_matrix[i, j]
                if sim > 0.95:
                    marker = "⚠️"
                elif sim > 0.90:
                    marker = "⚠ "
                elif sim < 0.70:
                    marker = "✓ "
                else:
                    marker = "  "
                row += f"{marker}{sim:>13.4f}"
        lines.append(row)

    lines.append(f"\n{'─'*70}")
    lines.append("Euclidean Distance (higher = more diverged)")
    lines.append(f"{'─'*70}")

    header = f"{'':20}"
    for name in prompt_names:
        header += f"{name[:15]:>15}"
    lines.append(header)

    for i, name_i in enumerate(prompt_names):
        row = f"{name_i:20}"
        for j in range(n):
            if i == j:
                row += f"{'0.0000':>15}"
            else:
                dist = euclidean_matrix[i, j]
                if dist < 1.0:
                    marker = "⚠️"
                elif dist < 2.0:
                    marker = "⚠ "
                elif dist > 5.0:
                    marker = "✓ "
                else:
                    marker = "  "
                row += f"{marker}{dist:>13.4f}"
        lines.append(row)

    return "\n".join(lines)


def _interpret_divergence(results):
    avg_sim = results["avg_sim"]
    ckpt = results["ckpt"]

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("INTERPRETATION")
    lines.append(f"{'='*70}\n")

    if avg_sim > 0.95:
        lines.append("❌ CRITICAL PROBLEM: Prompts are nearly identical!")
        lines.append("   Average cosine similarity > 0.95 indicates prompts have NOT diverged.\n")
        lines.append("   Root causes:")
        lines.append("   • Learning rate too low → prompts can't move away from initialization")
        lines.append("   • Weight decay too high → regularization prevents divergence")
        lines.append("   • Initialization noise too small → prompts start too close together")
        lines.append("   • Prompts are frozen or not receiving gradients\n")
        lines.append("   Recommended fixes:")
        lines.append(f"   1. Increase learning rate: 1e-4 or 2e-4 (currently using {ckpt.get('config', {}).get('training', {}).get('lr', 'unknown')})")
        lines.append("   2. Increase initialization noise: 0.15-0.25 (in train.py)")
        lines.append("   3. Reduce weight decay temporarily: 5e-3")
        lines.append("   4. Check that prompts have requires_grad=True")
        lines.append("   5. Verify gradients are flowing to prompt parameters")

    elif avg_sim > 0.90:
        lines.append("⚠️  WARNING: Prompts are quite similar")
        lines.append("   Average cosine similarity > 0.90 indicates insufficient divergence.\n")
        lines.append("   Prompts have started to separate but need more training or stronger signals.\n")
        lines.append("   Recommended adjustments:")
        lines.append(f"   1. Increase learning rate: 1e-4 (currently using {ckpt.get('config', {}).get('training', {}).get('lr', 'unknown')})")
        lines.append("   2. Increase initialization noise: 0.10-0.15 (in train.py)")
        lines.append("   3. Continue training for more epochs")
        lines.append("   4. Consider class-specific data augmentation")

    elif avg_sim > 0.80:
        lines.append("⚠  MODERATE: Prompts show some divergence")
        lines.append("   Average cosine similarity 0.80-0.90 indicates partial separation.\n")
        lines.append("   This may be acceptable depending on:")
        lines.append("   • How acoustically similar your classes are (airplane vs birds = quite different)")
        lines.append("   • How many training epochs have run")
        lines.append("   • Whether separation metrics are good enough\n")
        lines.append("   Consider:")
        lines.append("   • Continuing training to see if divergence improves")
        lines.append("   • Checking per-class separation performance")

    elif avg_sim > 0.70:
        lines.append("✓ ACCEPTABLE: Prompts have moderate divergence")
        lines.append("   Average cosine similarity 0.70-0.80 indicates reasonable separation.\n")
        lines.append("   Classes are distinguishable but share some common features.")
        lines.append("   This is often appropriate when classes have acoustic overlap.")

    else:
        lines.append("✅ EXCELLENT: Prompts have diverged well!")
        lines.append("   Average cosine similarity < 0.70 indicates strong separation.\n")
        lines.append("   Prompts have learned distinct representations for each class.")
        lines.append("   This suggests good class discrimination capability.")

    return "\n".join(lines)


def analyze_prompt_divergence(checkpoint_path: str | Path, target_prompts: list[str] | None = None):
    """Analyze how much prompts have diverged from each other.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        target_prompts: Optional list of specific prompts to analyze. If None, analyzes all.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    results = _load_and_compute_divergence(checkpoint_path, target_prompts)
    if results is None:
        return

    print(f"\n{'='*70}")
    print(f"PROMPT DIVERGENCE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nAnalyzing {results['n']} prompts:")
    for name in results["prompt_names"]:
        vec = results["prompts"][name]
        norm = torch.norm(vec.flatten()).item()
        print(f"  {name:20s} shape: {tuple(vec.shape)}, L2 norm: {norm:.4f}")

    print(_format_divergence_tables(results))

    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    similarities = results["similarities"]
    if similarities:
        print(f"\nPairwise Comparisons ({results['n_pairs']} pairs):")
        print(f"  Cosine Similarity:")
        print(f"    Mean:  {results['avg_sim']:.4f} ± {results['std_sim']:.4f}")
        print(f"    Range: [{results['min_sim']:.4f}, {results['max_sim']:.4f}]")
        print(f"\n  Euclidean Distance:")
        print(f"    Mean:  {results['avg_dist']:.4f} ± {results['std_dist']:.4f}")
        print(f"    Range: [{results['min_dist']:.4f}, {results['max_dist']:.4f}]")

        print(f"\nDetailed Pairwise Analysis:")
        for name_i, name_j, sim, dist in results["pairwise_info"]:
            status = ""
            if sim > 0.95:
                status = "❌ TOO SIMILAR"
            elif sim > 0.90:
                status = "⚠️  WARNING"
            elif sim < 0.70:
                status = "✓ GOOD"

            print(f"  {name_i:15s} <-> {name_j:15s}  |  cos: {sim:6.4f}  |  dist: {dist:6.4f}  {status}")

    print(_interpret_divergence(results))

    ckpt = results["ckpt"]
    print(f"\n{'─'*70}")
    print("Checkpoint Training Info:")
    print(f"{'─'*70}")
    config = ckpt.get("config", {})
    training_cfg = config.get("training", {})
    print(f"  Learning rate: {training_cfg.get('lr', 'unknown')}")
    print(f"  Weight decay: {training_cfg.get('weight_decay', 'unknown')}")
    print(f"  Epochs trained: {ckpt.get('epoch', 'unknown')}")
    print(f"  Global steps: {ckpt.get('global_step', 'unknown')}")
    print(f"  Best val loss: {ckpt.get('val_loss', 'unknown')}")

    print(f"\n{'='*70}\n")


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
    parser.add_argument(
        "-d", "--divergence",
        action="store_true",
        help="Analyze prompt divergence (cosine similarity and Euclidean distance)",
    )
    parser.add_argument(
        "-p", "--prompts",
        type=str,
        nargs="+",
        metavar="PROMPT",
        help="Specific prompts to analyze (space-separated). If not specified, analyzes all prompts.",
    )
    
    args = parser.parse_args()
    
    try:
        if args.divergence:
            analyze_prompt_divergence(args.checkpoint, target_prompts=args.prompts)
        elif args.compare_config:
            compare_with_config(args.checkpoint, args.compare_config)
        else:
            print_checkpoint_info(args.checkpoint, verbose=args.verbose)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
