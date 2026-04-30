"""Analyze prompt vector divergence in a TUSS checkpoint.

This script loads a trained checkpoint and computes:
1. Cosine similarity between all prompt pairs
2. L2 distances between prompts
3. Visualization of prompt relationships
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
sys.path.insert(0, "/home/bendm/Thesis/project/code/src")


def load_checkpoint_prompts(ckpt_path: Path):
    """Load prompt vectors from a checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Find the checkpoint file
    if ckpt_path.is_dir():
        ckpt_file = ckpt_path / "best_model.pt"
        if not ckpt_file.exists():
            ckpt_files = list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth"))
            if ckpt_files:
                ckpt_file = ckpt_files[0]
            else:
                raise FileNotFoundError(f"No checkpoint file found in {ckpt_path}")
    else:
        ckpt_file = ckpt_path
    
    print(f"  Loading: {ckpt_file}")
    
    try:
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    except Exception as e:
        if "numpy._core.multiarray.scalar" in str(e):
            import numpy as _np
            from torch.serialization import safe_globals
            with safe_globals([_np._core.multiarray.scalar]):
                checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        else:
            raise
    
    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        coi_prompts = checkpoint.get("coi_prompts", [])
        bg_prompt = checkpoint.get("bg_prompt", "")
        print(f"  Checkpoint prompts: COI={coi_prompts}, BG={bg_prompt}")
    else:
        state_dict = checkpoint
        coi_prompts = []
        bg_prompt = ""
    
    # Extract prompt vectors
    prompt_vectors = {}
    prompt_prefix = "separator.prompts."
    
    for key, value in state_dict.items():
        if key.startswith(prompt_prefix):
            prompt_name = key[len(prompt_prefix):]
            prompt_vectors[prompt_name] = value.cpu().numpy()
            print(f"  Found prompt: {prompt_name} with shape {value.shape}")
    
    return prompt_vectors, coi_prompts, bg_prompt


def compute_similarity_metrics(prompt_vectors: dict):
    """Compute cosine similarity and L2 distance between all prompt pairs."""
    prompt_names = list(prompt_vectors.keys())
    n_prompts = len(prompt_names)
    
    # Initialize matrices
    cos_sim_matrix = np.zeros((n_prompts, n_prompts))
    l2_dist_matrix = np.zeros((n_prompts, n_prompts))
    
    for i, name_i in enumerate(prompt_names):
        vec_i = prompt_vectors[name_i].flatten()
        for j, name_j in enumerate(prompt_names):
            vec_j = prompt_vectors[name_j].flatten()
            
            # Cosine similarity
            cos_sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)
            cos_sim_matrix[i, j] = cos_sim
            
            # L2 distance
            l2_dist = np.linalg.norm(vec_i - vec_j)
            l2_dist_matrix[i, j] = l2_dist
    
    return cos_sim_matrix, l2_dist_matrix, prompt_names


def print_analysis(cos_sim_matrix, l2_dist_matrix, prompt_names):
    """Print detailed analysis of prompt relationships."""
    n = len(prompt_names)
    
    print("\n" + "="*80)
    print("PROMPT DIVERGENCE ANALYSIS")
    print("="*80)
    
    print("\n--- Cosine Similarity Matrix ---")
    print("(1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
    print("\n", end="")
    
    # Print header
    max_name_len = max(len(name) for name in prompt_names)
    print(f"{'':>{max_name_len}}", end="")
    for name in prompt_names:
        print(f"  {name[:8]:>8}", end="")
    print()
    
    # Print rows
    for i, name_i in enumerate(prompt_names):
        print(f"{name_i:>{max_name_len}}", end="")
        for j in range(n):
            print(f"  {cos_sim_matrix[i, j]:>8.4f}", end="")
        print()
    
    print("\n--- L2 Distance Matrix ---")
    print("(0.0 = identical, larger = more different)")
    print("\n", end="")
    
    # Print header
    print(f"{'':>{max_name_len}}", end="")
    for name in prompt_names:
        print(f"  {name[:8]:>8}", end="")
    print()
    
    # Print rows
    for i, name_i in enumerate(prompt_names):
        print(f"{name_i:>{max_name_len}}", end="")
        for j in range(n):
            print(f"  {l2_dist_matrix[i, j]:>8.2f}", end="")
        print()
    
    print("\n--- Pairwise Analysis ---")
    pairs_analyzed = set()
    for i, name_i in enumerate(prompt_names):
        for j, name_j in enumerate(prompt_names):
            if i < j:  # Only analyze each pair once
                pair_key = tuple(sorted([name_i, name_j]))
                if pair_key not in pairs_analyzed:
                    pairs_analyzed.add(pair_key)
                    cos_sim = cos_sim_matrix[i, j]
                    l2_dist = l2_dist_matrix[i, j]
                    
                    # Interpret results
                    if cos_sim > 0.95:
                        interpretation = "⚠️  VERY SIMILAR (poor separation expected)"
                    elif cos_sim > 0.85:
                        interpretation = "⚠️  Similar (weak separation)"
                    elif cos_sim > 0.70:
                        interpretation = "⚡ Moderate similarity (acceptable)"
                    elif cos_sim > 0.50:
                        interpretation = "✅ Good divergence"
                    else:
                        interpretation = "✅ Strong divergence"
                    
                    print(f"\n{name_i} ↔ {name_j}:")
                    print(f"  Cosine similarity: {cos_sim:.4f}")
                    print(f"  L2 distance: {l2_dist:.2f}")
                    print(f"  {interpretation}")
    
    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    # Get off-diagonal cosine similarities (excluding self-similarity)
    off_diag_sims = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag_sims.append(cos_sim_matrix[i, j])
    
    avg_sim = np.mean(off_diag_sims)
    max_sim = np.max(off_diag_sims)
    min_sim = np.min(off_diag_sims)
    
    print(f"\nAverage inter-prompt similarity: {avg_sim:.4f}")
    print(f"Max inter-prompt similarity: {max_sim:.4f}")
    print(f"Min inter-prompt similarity: {min_sim:.4f}")
    
    if avg_sim > 0.90:
        print("\n⚠️  WARNING: Prompts are VERY similar!")
        print("   → Model likely cannot distinguish between classes well")
        print("   → Background separation will be poor")
        print("   → Recommendation: Train longer or increase learning rate for prompts")
    elif avg_sim > 0.75:
        print("\n⚠️  Prompts have moderate similarity")
        print("   → Some separation possible but not optimal")
        print("   → Background may struggle to separate cleanly")
    else:
        print("\n✅ Prompts are well-differentiated")
        print("   → Model should be able to separate sources effectively")


def plot_similarity_heatmap(cos_sim_matrix, prompt_names, save_path: Path):
    """Create a heatmap visualization of prompt similarities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cosine similarity heatmap
    sns.heatmap(
        cos_sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        xticklabels=prompt_names,
        yticklabels=prompt_names,
        ax=ax1,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax1.set_title("Prompt Cosine Similarity\n(lower = better separation)")
    
    # Angle in degrees (derived from cosine similarity)
    angle_matrix = np.arccos(np.clip(cos_sim_matrix, -1, 1)) * 180 / np.pi
    sns.heatmap(
        angle_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=180,
        xticklabels=prompt_names,
        yticklabels=prompt_names,
        ax=ax2,
        cbar_kws={"label": "Angle (degrees)"},
    )
    ax2.set_title("Prompt Vector Angles\n(higher = better separation)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()


def main():
    ckpt_path = Path(
        "/home/bendm/Thesis/project/code/src/models/tuss/checkpoints/multi_coi_29_04"
    )
    
    # Load prompts
    prompt_vectors, coi_prompts, bg_prompt = load_checkpoint_prompts(ckpt_path)
    
    if not prompt_vectors:
        print("❌ No prompt vectors found in checkpoint!")
        return
    
    # Compute metrics
    cos_sim_matrix, l2_dist_matrix, prompt_names = compute_similarity_metrics(
        prompt_vectors
    )
    
    # Print analysis
    print_analysis(cos_sim_matrix, l2_dist_matrix, prompt_names)
    
    # Plot heatmap
    save_path = Path(
        "/home/bendm/Thesis/project/code/src/validation_functions/demo_output/prompt_similarity_heatmap.png"
    )
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plot_similarity_heatmap(cos_sim_matrix, prompt_names, save_path)


if __name__ == "__main__":
    main()
