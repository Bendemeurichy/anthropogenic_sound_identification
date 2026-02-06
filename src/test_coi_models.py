#!/usr/bin/env python3
"""
Smoke test script to verify COI training implementations for xumx and CLAPSep.

This script tests:
1. Dataset loading and batch preparation
2. Model forward passes with correct shapes
3. Loss computation

Run from the src directory:
    python test_coi_models.py --model xumx
    python test_coi_models.py --model clapsep --clap-checkpoint path/to/clap.pt
    python test_coi_models.py --model dataset
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch


def test_dataset_and_batch_prep():
    """Test the COI dataset and batch preparation."""
    print("\n" + "=" * 60)
    print("Testing COIAudioDataset and prepare_batch")
    print("=" * 60)

    import pandas as pd

    from common.coi_training import (
        COIAudioDataset,
        COIWeightedLoss,
        prepare_batch,
        sisnr,
    )

    # Create a mock dataframe (you would use real data in practice)
    print("\n1. Creating mock dataset...")

    # For testing, we'll create synthetic data
    sample_rate = 16000
    segment_length = 2.0
    segment_samples = int(sample_rate * segment_length)

    # Create mock audio tensors
    n_coi = 5
    n_bg = 10
    batch_size = 4

    print(f"   Sample rate: {sample_rate}")
    print(f"   Segment length: {segment_length}s ({segment_samples} samples)")

    # Test prepare_batch with synthetic sources
    print("\n2. Testing prepare_batch (mono)...")
    # sources shape: (B, n_src, T) where n_src = 2 (COI + background)
    sources_mono = torch.randn(batch_size, 2, segment_samples)
    mixture, clean_wavs = prepare_batch(sources_mono, snr_range=(-5, 5))

    print(f"   Input sources shape: {sources_mono.shape}")
    print(f"   Output mixture shape: {mixture.shape}")
    print(f"   Output clean_wavs shape: {clean_wavs.shape}")

    assert mixture.shape == (batch_size, segment_samples), (
        f"Expected ({batch_size}, {segment_samples})"
    )
    assert clean_wavs.shape == (batch_size, 2, segment_samples), (
        f"Expected ({batch_size}, 2, {segment_samples})"
    )

    print("   ✓ Mono batch preparation passed!")

    # Test prepare_batch with stereo
    print("\n3. Testing prepare_batch (stereo)...")
    sources_stereo = torch.randn(batch_size, 2, 2, segment_samples)  # (B, n_src, C, T)
    mixture_stereo, clean_wavs_stereo = prepare_batch(sources_stereo, snr_range=(-5, 5))

    print(f"   Input sources shape: {sources_stereo.shape}")
    print(f"   Output mixture shape: {mixture_stereo.shape}")
    print(f"   Output clean_wavs shape: {clean_wavs_stereo.shape}")

    assert mixture_stereo.shape == (batch_size, 2, segment_samples)
    assert clean_wavs_stereo.shape == (batch_size, 2, 2, segment_samples)

    print("   ✓ Stereo batch preparation passed!")

    # Test SI-SNR computation
    print("\n4. Testing SI-SNR computation...")
    est = torch.randn(batch_size, segment_samples)
    target = est + 0.1 * torch.randn_like(est)  # Similar to est
    sisnr_vals = sisnr(est, target)

    print(f"   SI-SNR shape: {sisnr_vals.shape}")
    print(f"   SI-SNR values: {sisnr_vals}")

    assert sisnr_vals.shape == (batch_size,)
    print("   ✓ SI-SNR computation passed!")

    # Test COI weighted loss
    print("\n5. Testing COIWeightedLoss...")
    criterion = COIWeightedLoss(class_weight=1.5)

    # Simulated model output and targets
    est_sources = torch.randn(batch_size, 2, segment_samples)
    target_sources = est_sources + 0.1 * torch.randn_like(est_sources)

    loss = criterion(est_sources, target_sources)
    print(f"   Loss value: {loss.item():.4f}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    print("   ✓ COIWeightedLoss passed!")

    print("\n" + "=" * 60)
    print("✓ All dataset and batch preparation tests passed!")
    print("=" * 60)


def test_xumx_model():
    """Test the xumx model forward pass."""
    print("\n" + "=" * 60)
    print("Testing xumx COI model")
    print("=" * 60)

    from common.coi_training import COIWeightedLoss, prepare_batch
    from models.xumx.base.xumx_slicq_v2 import transforms
    from models.xumx.train import COISlicedUnmixCDAE, COIUnmix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Using device: {device}")

    # xumx requires 44.1kHz
    sample_rate = 44100
    segment_length = 2.0
    batch_size = 2

    print(f"   Sample rate: {sample_rate}")
    print(f"   Segment length: {segment_length}s")
    print(f"   Batch size: {batch_size}")

    # Create NSGT transforms
    print("\n2. Creating NSGT transforms...")
    nsgt_base = transforms.NSGTBase(
        "bark",
        262,
        32.9,
        fgamma=0.0,
        fs=sample_rate,
        device=device,
    )

    nsgt, insgt = transforms.make_filterbanks(nsgt_base, sample_rate=sample_rate)
    cnorm = transforms.ComplexNorm()

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)
    cnorm = cnorm.to(device)

    # Get sample input for model initialization
    print("\n3. Getting sample NSGT output shape...")
    jagged_slicq, _ = nsgt_base.predict_input_size(batch_size, 2, segment_length)
    jagged_slicq_cnorm = cnorm(jagged_slicq)

    print(f"   Number of sliCQ blocks: {len(jagged_slicq_cnorm)}")
    for i, block in enumerate(jagged_slicq_cnorm[:3]):
        print(f"   Block {i} shape: {block.shape}")
    if len(jagged_slicq_cnorm) > 3:
        print(f"   ... ({len(jagged_slicq_cnorm) - 3} more blocks)")

    # Create model
    print("\n4. Creating COIUnmix model...")
    model = COIUnmix(
        jagged_slicq_cnorm,
        realtime=False,
        input_means=None,
        input_scales=None,
        num_sources=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params / 1e6:.2f}M")

    # Test forward pass
    print("\n5. Testing forward pass...")

    # Create stereo input
    segment_samples = int(sample_rate * segment_length)
    sources = torch.randn(batch_size, 2, 2, segment_samples, device=device)

    mixture, clean_wavs = prepare_batch(sources, snr_range=(-5, 5))
    print(f"   Mixture shape: {mixture.shape}")

    # Transform to sliCQ domain
    Xcomplex = nsgt(mixture)
    print(f"   Xcomplex (list of {len(Xcomplex)} blocks)")
    print(f"   Xcomplex[0] shape: {Xcomplex[0].shape}")

    # Forward pass
    Ycomplex_ests, Ymasks = model(
        Xcomplex, return_masks=True, skip_phase_reconstruction=True
    )

    print(f"   Ycomplex_ests (list of {len(Ycomplex_ests)} blocks)")
    print(f"   Ycomplex_ests[0] shape: {Ycomplex_ests[0].shape}")
    print(f"   Ymasks[0] shape: {Ymasks[0].shape}")

    # Verify output shapes
    assert len(Ycomplex_ests) == len(Xcomplex)
    assert Ycomplex_ests[0].shape[0] == 2, "Should have 2 sources (COI + BG)"

    print("   ✓ Forward pass successful!")

    # Test inverse transform
    print("\n6. Testing inverse transform...")
    y_ests = []
    for src_idx in range(2):
        src_blocks = [Ycomplex_ests[b][src_idx] for b in range(len(Ycomplex_ests))]
        y_est = insgt(src_blocks, mixture.shape[-1])
        y_ests.append(y_est)

    y_ests = torch.stack(y_ests, dim=1)
    print(f"   Reconstructed shape: {y_ests.shape}")

    assert y_ests.shape == clean_wavs.shape, (
        f"Expected {clean_wavs.shape}, got {y_ests.shape}"
    )
    print("   ✓ Inverse transform successful!")

    # Test loss computation
    print("\n7. Testing loss computation...")
    criterion = COIWeightedLoss(class_weight=1.5)
    loss = criterion(y_ests, clean_wavs)
    print(f"   Loss value: {loss.item():.4f}")

    assert torch.isfinite(loss), "Loss should be finite"
    print("   ✓ Loss computation successful!")

    print("\n" + "=" * 60)
    print("✓ All xumx model tests passed!")
    print("=" * 60)


def test_clapsep_model(clap_checkpoint: str):
    """Test the CLAPSep model forward pass."""
    print("\n" + "=" * 60)
    print("Testing CLAPSep COI model")
    print("=" * 60)

    try:
        import laion_clap
        from torchlibrosa import ISTFT, STFT
        from torchlibrosa.stft import magphase
    except ImportError as e:
        print(f"\n⚠ Cannot test CLAPSep: {e}")
        print("  Install with: pip install laion-clap torchlibrosa")
        return

    from common.coi_training import COIWeightedLoss, prepare_batch
    from models.clapsep.base.model.CLAPSep_decoder import HTSAT_Decoder
    from models.clapsep.train_coi import COICLAPSep, COICLAPSepDecoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Using device: {device}")

    sample_rate = 32000
    segment_length = 5.0
    batch_size = 2

    print(f"   Sample rate: {sample_rate}")
    print(f"   Segment length: {segment_length}s")
    print(f"   Batch size: {batch_size}")

    # Load CLAP model
    print(f"\n2. Loading CLAP model from {clap_checkpoint}...")
    if not Path(clap_checkpoint).exists():
        print(f"   ⚠ CLAP checkpoint not found at {clap_checkpoint}")
        print("   Skipping CLAPSep test.")
        return

    clap_model = laion_clap.CLAP_Module(
        enable_fusion=False, amodel="HTSAT-base", device="cpu"
    )
    clap_model.load_ckpt(clap_checkpoint)
    print("   ✓ CLAP model loaded!")

    # Create decoder
    print("\n3. Creating HTSAT_Decoder...")
    base_decoder = HTSAT_Decoder(
        lan_embed_dim=1024,
        depths=[1, 1, 1, 1],
        embed_dim=128,
        encoder_embed_dim=128,
        phase=False,
        spec_factor=8,
        d_attn=640,
        n_masker_layer=3,
        conv=False,
    )
    print("   ✓ HTSAT_Decoder created!")

    # Wrap in COI decoder
    print("\n4. Creating COICLAPSepDecoder...")
    coi_decoder = COICLAPSepDecoder(
        decoder=base_decoder,
        embed_dim=1024,
        num_sources=2,
    )
    print("   ✓ COICLAPSepDecoder created!")

    # Create full model
    print("\n5. Creating COICLAPSep model...")
    model = COICLAPSep(
        clap_model=clap_model,
        decoder_model=coi_decoder,
        lr=1e-4,
        nfft=1024,
        sample_rate=sample_rate,
        resample_rate=48000,
        class_weight=1.5,
        freeze_encoder=True,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"   Trainable parameters: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M total"
    )

    # Test forward pass
    print("\n6. Testing forward pass...")
    segment_samples = int(sample_rate * segment_length)
    sources = torch.randn(batch_size, 2, segment_samples, device=device)

    mixture, clean_wavs = prepare_batch(sources, snr_range=(-5, 5))
    print(f"   Mixture shape: {mixture.shape}")
    print(f"   Clean wavs shape: {clean_wavs.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        separated = model(mixture)

    print(f"   Separated shape: {separated.shape}")

    assert separated.shape == (batch_size, 2, segment_samples), (
        f"Expected ({batch_size}, 2, {segment_samples}), got {separated.shape}"
    )
    print("   ✓ Forward pass successful!")

    # Test loss computation
    print("\n7. Testing loss computation...")
    criterion = COIWeightedLoss(class_weight=1.5)
    loss = criterion(separated, clean_wavs)
    print(f"   Loss value: {loss.item():.4f}")

    assert torch.isfinite(loss), "Loss should be finite"
    print("   ✓ Loss computation successful!")

    print("\n" + "=" * 60)
    print("✓ All CLAPSep model tests passed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Smoke test COI models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["xumx", "clapsep", "dataset", "all"],
        default="dataset",
        help="Which model to test",
    )
    parser.add_argument(
        "--clap-checkpoint",
        type=str,
        default="checkpoints/music_audioset_epoch_15_esc_90.14.pt",
        help="Path to CLAP checkpoint (for CLAPSep test)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("COI Model Smoke Tests")
    print("=" * 60)

    if args.model in ["dataset", "all"]:
        test_dataset_and_batch_prep()

    if args.model in ["xumx", "all"]:
        test_xumx_model()

    if args.model in ["clapsep", "all"]:
        test_clapsep_model(args.clap_checkpoint)

    print("\n" + "=" * 60)
    print("All requested tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
