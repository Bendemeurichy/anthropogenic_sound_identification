"""Analyze the separation outputs from demo."""
import sys
sys.path.insert(0, '/home/bendm/Thesis/project/code/src')

import torch
import soundfile as sf
import numpy as np

output_dir = '/home/bendm/Thesis/project/code/src/validation_functions/demo_output'

# Load all files
mixture, sr = sf.read(f'{output_dir}/mixture.wav')
plane, _ = sf.read(f'{output_dir}/separated_plane.wav')
bird, _ = sf.read(f'{output_dir}/separated_bird.wav')
bg, _ = sf.read(f'{output_dir}/separated_background.wav')

print("="*70)
print("SEPARATION OUTPUT ANALYSIS")
print("="*70)

print(f"\nSample rate: {sr} Hz")
print(f"Duration: {len(mixture) / sr:.2f}s ({len(mixture)} samples)")

print(f"\nEnergy analysis:")
mixture_energy = np.sum(mixture**2)
plane_energy = np.sum(plane**2)
bird_energy = np.sum(bird**2)
bg_energy = np.sum(bg**2)
total_separated_energy = plane_energy + bird_energy + bg_energy

print(f"  Mixture:    {mixture_energy:.2e} (100%)")
print(f"  Plane:      {plane_energy:.2e} ({100 * plane_energy / mixture_energy:.2f}%)")
print(f"  Bird:       {bird_energy:.2e} ({100 * bird_energy / mixture_energy:.2f}%)")
print(f"  Background: {bg_energy:.2e} ({100 * bg_energy / mixture_energy:.2f}%)")
print(f"  Total separated: {total_separated_energy:.2e} ({100 * total_separated_energy / mixture_energy:.2f}%)")

# Check if background is just the mixture
mixture_norm = mixture / (np.sqrt(mixture_energy) + 1e-8)
bg_norm = bg / (np.sqrt(bg_energy) + 1e-8)
correlation = np.sum(mixture_norm * bg_norm)
print(f"\nCorrelation between mixture and background: {correlation:.4f}")
print(f"  (1.0 = identical, 0.0 = uncorrelated)")

# Check similarity between plane and bird outputs
plane_norm = plane / (np.sqrt(plane_energy) + 1e-8)
bird_norm = bird / (np.sqrt(bird_energy) + 1e-8)
plane_bird_corr = np.sum(plane_norm * bird_norm)
print(f"\nCorrelation between plane and bird outputs: {plane_bird_corr:.4f}")
print(f"  (High correlation = model can't distinguish classes)")

# Check peak amplitudes
print(f"\nPeak amplitudes:")
print(f"  Mixture:    {np.abs(mixture).max():.4f}")
print(f"  Plane:      {np.abs(plane).max():.4f}")
print(f"  Bird:       {np.abs(bird).max():.4f}")
print(f"  Background: {np.abs(bg).max():.4f}")

# Reconstruction check
reconstructed = plane + bird + bg
reconstruction_error = np.sum((mixture - reconstructed)**2)
print(f"\nReconstruction check:")
print(f"  Error: {reconstruction_error:.2e}")
print(f"  SNR: {10 * np.log10(mixture_energy / (reconstruction_error + 1e-10)):.2f} dB")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
if correlation > 0.95:
    print("❌ CRITICAL: Background is nearly identical to mixture!")
    print("   The model is NOT separating - just copying input to background.")
if plane_bird_corr > 0.95:
    print("❌ CRITICAL: Plane and bird outputs are nearly identical!")
    print("   The model can't distinguish between COI classes.")
if plane_energy < 0.01 * mixture_energy:
    print("❌ CRITICAL: Plane output has very little energy!")
    print("   The model is not extracting the airplane.")
if bird_energy < 0.01 * mixture_energy:
    print("❌ CRITICAL: Bird output has very little energy!")
    print("   The model is not extracting the birds.")

print("\n" + "="*70)
