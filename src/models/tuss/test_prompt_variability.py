#!/usr/bin/env python3
"""
Test script for prompt variability implementation.

Tests:
1. generate_variable_prompts - random prompt generation
2. select_sources_for_prompts - ground truth channel selection
3. COIWeightedSNRLoss - dynamic n_src handling
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from train import generate_variable_prompts, select_sources_for_prompts, COIWeightedSNRLoss


def test_generate_variable_prompts():
    """Test variable prompt generation."""
    print("=" * 70)
    print("Test 1: generate_variable_prompts")
    print("=" * 70)
    
    coi_prompts = ["airplane", "birds"]
    bg_prompt = "background"
    rng = np.random.default_rng(42)
    
    # Test 1: Generate 5 samples with 50% dropout, min_coi=0
    print("\n1. Generate 5 samples (dropout_prob=0.5, min_coi=0):")
    prompts = generate_variable_prompts(
        coi_prompts, bg_prompt, batch_size=5,
        dropout_prob=0.5, min_coi=0, rng=rng
    )
    for i, p in enumerate(prompts):
        print(f"   Sample {i}: {p}")
    
    # Test 2: min_coi=1 ensures at least 1 COI prompt
    print("\n2. Generate 5 samples (dropout_prob=0.5, min_coi=1):")
    rng = np.random.default_rng(42)
    prompts = generate_variable_prompts(
        coi_prompts, bg_prompt, batch_size=5,
        dropout_prob=0.5, min_coi=1, rng=rng
    )
    for i, p in enumerate(prompts):
        print(f"   Sample {i}: {p}")
        assert len(p) >= 2, f"Sample {i} should have at least 1 COI + 1 BG"
    
    # Test 3: dropout_prob=0.9 (rarely keep prompts)
    print("\n3. Generate 5 samples (dropout_prob=0.9, min_coi=0):")
    rng = np.random.default_rng(42)
    prompts = generate_variable_prompts(
        coi_prompts, bg_prompt, batch_size=5,
        dropout_prob=0.9, min_coi=0, rng=rng
    )
    for i, p in enumerate(prompts):
        print(f"   Sample {i}: {p}")
    
    print("\n✅ Test 1 PASSED\n")


def test_select_sources_for_prompts():
    """Test ground truth channel selection."""
    print("=" * 70)
    print("Test 2: select_sources_for_prompts")
    print("=" * 70)
    
    coi_prompts = ["airplane", "birds"]
    bg_prompt = "background"
    
    # Create dummy clean_wavs: (B=2, n_src=3, T=100)
    B, T = 2, 100
    clean_wavs = torch.randn(B, 3, T)
    # Mark each channel with distinct value for verification
    clean_wavs[:, 0, :] = 1.0  # airplane
    clean_wavs[:, 1, :] = 2.0  # birds
    clean_wavs[:, 2, :] = 3.0  # background
    
    print(f"\nInput clean_wavs shape: {clean_wavs.shape}")
    print(f"Channel 0 (airplane): mean={clean_wavs[:, 0].mean():.1f}")
    print(f"Channel 1 (birds): mean={clean_wavs[:, 1].mean():.1f}")
    print(f"Channel 2 (background): mean={clean_wavs[:, 2].mean():.1f}")
    
    # Test 1: Select all prompts (original order)
    print("\n1. Selected prompts: ['airplane', 'birds', 'background']")
    selected = select_sources_for_prompts(
        clean_wavs, coi_prompts, bg_prompt,
        ["airplane", "birds", "background"]
    )
    print(f"   Output shape: {selected.shape}")
    print(f"   Channel 0 mean: {selected[:, 0].mean():.1f} (expect 1.0)")
    print(f"   Channel 1 mean: {selected[:, 1].mean():.1f} (expect 2.0)")
    print(f"   Channel 2 mean: {selected[:, 2].mean():.1f} (expect 3.0)")
    assert selected.shape == (B, 3, T)
    assert selected[:, 0].mean().item() == 1.0
    assert selected[:, 1].mean().item() == 2.0
    assert selected[:, 2].mean().item() == 3.0
    
    # Test 2: Select birds + background (reordered)
    print("\n2. Selected prompts: ['birds', 'background']")
    selected = select_sources_for_prompts(
        clean_wavs, coi_prompts, bg_prompt,
        ["birds", "background"]
    )
    print(f"   Output shape: {selected.shape}")
    print(f"   Channel 0 mean: {selected[:, 0].mean():.1f} (expect 2.0 = birds)")
    print(f"   Channel 1 mean: {selected[:, 1].mean():.1f} (expect 3.0 = background)")
    assert selected.shape == (B, 2, T)
    assert selected[:, 0].mean().item() == 2.0  # birds
    assert selected[:, 1].mean().item() == 3.0  # background
    
    # Test 3: Select only background
    print("\n3. Selected prompts: ['background']")
    selected = select_sources_for_prompts(
        clean_wavs, coi_prompts, bg_prompt,
        ["background"]
    )
    print(f"   Output shape: {selected.shape}")
    print(f"   Channel 0 mean: {selected[:, 0].mean():.1f} (expect 3.0 = background)")
    assert selected.shape == (B, 1, T)
    assert selected[:, 0].mean().item() == 3.0  # background
    
    # Test 4: Different order
    print("\n4. Selected prompts: ['background', 'airplane']")
    selected = select_sources_for_prompts(
        clean_wavs, coi_prompts, bg_prompt,
        ["background", "airplane"]
    )
    print(f"   Output shape: {selected.shape}")
    print(f"   Channel 0 mean: {selected[:, 0].mean():.1f} (expect 3.0 = background)")
    print(f"   Channel 1 mean: {selected[:, 1].mean():.1f} (expect 1.0 = airplane)")
    assert selected.shape == (B, 2, T)
    assert selected[:, 0].mean().item() == 3.0  # background first
    assert selected[:, 1].mean().item() == 1.0  # airplane second
    
    print("\n✅ Test 2 PASSED\n")


def test_coi_weighted_snr_loss_variable_nsrc():
    """Test loss function with variable n_src."""
    print("=" * 70)
    print("Test 3: COIWeightedSNRLoss with variable n_src")
    print("=" * 70)
    
    # Create loss function initialized for 3 sources
    criterion = COIWeightedSNRLoss(n_src=3, coi_weight=2.0)
    print(f"\nLoss initialized with n_src={criterion.n_src}, n_coi={criterion.n_coi}")
    
    B, T = 4, 4800
    
    # Test 1: Full 3 sources (2 COI + 1 BG)
    print("\n1. Testing with 3 sources (2 COI + 1 BG):")
    est = torch.randn(B, 3, T)
    ref = torch.randn(B, 3, T)
    loss = criterion(est, ref)
    print(f"   Input shapes: est={est.shape}, ref={ref.shape}")
    print(f"   Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"
    
    # Test 2: Reduced to 2 sources (1 COI + 1 BG)
    print("\n2. Testing with 2 sources (1 COI + 1 BG):")
    est = torch.randn(B, 2, T)
    ref = torch.randn(B, 2, T)
    loss = criterion(est, ref)
    print(f"   Input shapes: est={est.shape}, ref={ref.shape}")
    print(f"   Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"
    
    # Test 3: Minimal 1 source (only BG)
    print("\n3. Testing with 1 source (only BG):")
    est = torch.randn(B, 1, T)
    ref = torch.randn(B, 1, T)
    loss = criterion(est, ref)
    print(f"   Input shapes: est={est.shape}, ref={ref.shape}")
    print(f"   Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"
    
    # Test 4: With silent COI channels
    print("\n4. Testing with silent COI (airplane=0, birds=active, bg=active):")
    est = torch.randn(B, 3, T)
    ref = torch.randn(B, 3, T)
    ref[:, 0, :] = 0.0  # Airplane is silent
    loss = criterion(est, ref)
    print(f"   Input shapes: est={est.shape}, ref={ref.shape}")
    print(f"   Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN with silent channel"
    assert not torch.isinf(loss), "Loss should not be inf with silent channel"
    
    print("\n✅ Test 3 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PROMPT VARIABILITY IMPLEMENTATION TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_generate_variable_prompts()
        test_select_sources_for_prompts()
        test_coi_weighted_snr_loss_variable_nsrc()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nPrompt variability implementation is working correctly!")
        print("\nNext steps:")
        print("  1. Set variable_prompts: true in training_config.yaml")
        print("  2. Start training with: python train.py")
        print("  3. Monitor progress bar for variable class counts (c0, c1)")
        print("  4. At inference, you can now use any prompt subset:")
        print("     - model(mix, [['airplane', 'background']])")
        print("     - model(mix, [['birds', 'background']])")
        print("     - model(mix, [['airplane', 'birds', 'background']])")
        print("=" * 70 + "\n")
        return 0
    
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
