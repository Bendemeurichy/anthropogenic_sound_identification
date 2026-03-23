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


def load_wav_file(filepath: str, target_sr: int = None) -> torch.Tensor:
    """Load a WAV file and return as torch tensor."""
    sr, data = wavfile.read(filepath)
    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
    
    # Handle stereo -> mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    return torch.from_numpy(data), sr


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

    # Try loading a real audio file first
    test_audio_file = (
        Path(__file__).parent
        / "src/validation_functions/validation_examples_test/cnn/mixture_sep/mixture_created_0.wav"
    )
    
    if test_audio_file.exists():
        print(f"\n--- Loading real audio file ---")
        print(f"  File: {test_audio_file}")
        test_waveform, file_sr = load_wav_file(str(test_audio_file))
        print(f"  File sample rate: {file_sr}")
        print(f"  Loaded samples: {len(test_waveform)}")
        
        # Truncate/pad to segment_samples
        if len(test_waveform) > separator.segment_samples:
            test_waveform = test_waveform[:separator.segment_samples]
        elif len(test_waveform) < separator.segment_samples:
            test_waveform = torch.nn.functional.pad(
                test_waveform, (0, separator.segment_samples - len(test_waveform))
            )
    else:
        # Fallback: Create a test waveform (white noise)
        print(f"\n--- Creating synthetic test waveform ---")
        test_duration = 4.0  # seconds
        num_samples = int(separator.sample_rate * test_duration)
        test_waveform = torch.randn(num_samples) * 0.5

    print(f"\n--- Test Input ---")
    print(f"  Num samples: {len(test_waveform)}")
    print(f"  Duration: {len(test_waveform) / separator.sample_rate:.2f} s")
    print(f"  RMS energy: {torch.sqrt(torch.mean(test_waveform**2)):.6f}")
    print(f"  Peak amplitude: {test_waveform.abs().max():.6f}")
    
    # Debug: Check the normalization step
    mean = test_waveform.mean()
    std = test_waveform.std() + 1e-8
    normalized = (test_waveform - mean) / std
    print(f"  After normalization: mean={normalized.mean():.6f}, std={normalized.std():.6f}")

    # Run separation
    print(f"\n--- Running Separation ---")
    
    # Debug: Step through the separation manually
    print(f"\n--- Debug: Manual forward pass ---")
    segment = test_waveform
    mean = segment.mean()
    std = segment.std() + 1e-8
    
    # Normalize input (zero-mean, unit-variance) - matches training
    x = ((segment - mean) / std).unsqueeze(0).unsqueeze(0).to(separator.device)
    print(f"  Normalized input shape: {x.shape}")
    print(f"  Normalized input stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    print(f"  Input device: {x.device}")
    
    # Check model device placement
    model = separator.model
    model.eval()
    print(f"\n--- Debug: Model device placement ---")
    devices_found = set()
    for name, param in model.named_parameters():
        devices_found.add(str(param.device))
        if len(devices_found) > 1:
            print(f"  WARNING: Multiple devices detected!")
            break
    print(f"  Devices found in model parameters: {devices_found}")
    
    # Check specific layers
    if hasattr(model, 'encoder'):
        print(f"  encoder weight device: {model.encoder.weight.device}")
    if hasattr(model, 'decoder'):
        print(f"  decoder weight device: {model.decoder.weight.device}")
    if hasattr(model, 'mask_net'):
        for name, param in model.mask_net.named_parameters():
            print(f"  mask_net.{name} device: {param.device}")
            break  # Just show first one
    
    # Run model forward pass with gradient tracking for debugging
    with torch.inference_mode():
        estimates = model(x)
    
    print(f"\n--- Debug: Model output ---")
    print(f"  Raw model output shape: {estimates.shape}")
    print(f"  Raw model output device: {estimates.device}")
    print(f"  Raw model output stats: mean={estimates.mean():.6f}, std={estimates.std():.6f}, min={estimates.min():.6f}, max={estimates.max():.6f}")
    
    # Check each source in raw output
    for src_idx in range(estimates.shape[1]):
        src = estimates[0, src_idx]
        print(f"  Raw source {src_idx}: mean={src.mean():.6f}, std={src.std():.6f}, min={src.min():.6f}, max={src.max():.6f}")
    
    # Now use the actual separator method
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
