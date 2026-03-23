#!/usr/bin/env python3
"""
Diagnostic script to check if the SudoRM-RF model is outputting silence.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.sudormrf.inference import SeparationInference


def test_model_output():
    """Test if model outputs real signal or silence."""

    # Load the checkpoint
    checkpoint_path = (
        Path(__file__).parent
        / "src/models/sudormrf/checkpoints/silent_checkpoint/best_model.pt"
    )

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"✓ Loading checkpoint: {checkpoint_path}")
    separator = SeparationInference.from_checkpoint(
        str(checkpoint_path), device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"✓ Model loaded")
    print(f"  Sample rate: {separator.sample_rate}")
    print(f"  Segment samples: {separator.segment_samples}")
    print(f"  Device: {separator.device}")

    # Create a test waveform (white noise)
    test_duration = 4.0  # seconds
    num_samples = int(separator.sample_rate * test_duration)
    test_waveform = 

    print(f"\n--- Test Input ---")
    print(f"  Duration: {test_duration} s")
    print(f"  Num samples: {num_samples}")
    print(f"  RMS energy: {torch.sqrt(torch.mean(test_waveform**2)):.6f}")
    print(f"  Peak amplitude: {test_waveform.abs().max():.6f}")

    # Run separation
    print(f"\n--- Running Separation ---")
    with torch.inference_mode():
        separated = separator.separate_waveform(test_waveform)

    print(f"  Output shape: {separated.shape}")
    print(f"  Num sources: {separated.shape[0]}")

    # Check each output
    for src_idx in range(separated.shape[0]):
        src = separated[src_idx]
        rms = torch.sqrt(torch.mean(src**2))
        peak = src.abs().max()

        source_names = {0: "COI (Airplane)", 1: "Background"}
        name = source_names.get(src_idx, f"Source {src_idx}")

        print(f"\n  Source {src_idx} ({name}):")
        print(f"    RMS energy: {rms:.6f}")
        print(f"    Peak amplitude: {peak:.6f}")
        print(f"    Mean value: {src.mean():.6f}")
        print(f"    Std dev: {src.std():.6f}")

        # Check if output is silent
        if rms < 1e-6:
            print(f"    ⚠️  WARNING: Output is SILENT (RMS < 1e-6)")
        elif rms < 0.01:
            print(f"    ⚠️  WARNING: Output is VERY QUIET (RMS < 0.01)")
        else:
            print(f"    ✓ Output has reasonable energy")

    # Save test outputs using scipy
    output_dir = Path(__file__).parent / "test_separation_outputs"
    output_dir.mkdir(exist_ok=True)

    print(f"\n--- Saving Test Outputs ---")
    for src_idx in range(separated.shape[0]):
        output_path = output_dir / f"test_source_{src_idx}.wav"

        # Convert to numpy and scale to int16 range
        audio_np = separated[src_idx].cpu().numpy()

        # Normalize to prevent clipping
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            audio_np = audio_np / max_val * 0.95

        # Convert to int16
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Save with scipy
        wavfile.write(str(output_path), separator.sample_rate, audio_int16)
        print(f"  ✓ Saved: {output_path}")

    # Also save input for reference
    input_path = output_dir / "test_input.wav"
    input_np = test_waveform.cpu().numpy()

    # Normalize
    max_val = np.abs(input_np).max()
    if max_val > 0:
        input_np = input_np / max_val * 0.95

    # Convert to int16
    input_int16 = (input_np * 32767).astype(np.int16)
    wavfile.write(str(input_path), separator.sample_rate, input_int16)
    print(f"  ✓ Saved: {input_path}")

    print(f"\n✓ Test complete! Check files in {output_dir}")
    print(f"\nTo listen to the files:")
    print(f"  - test_input.wav: The input white noise")
    print(f"  - test_source_0.wav: COI (Airplane) channel output")
    print(f"  - test_source_1.wav: Background channel output")
    print(
        f"\nIf test_source_1.wav is nearly silent, the background suppression bug is confirmed."
    )


if __name__ == "__main__":
    test_model_output()
