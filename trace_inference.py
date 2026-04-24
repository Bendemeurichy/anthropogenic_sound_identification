"""Trace through the full inference pipeline with the demo audio."""
import sys
sys.path.insert(0, '/home/bendm/Thesis/project/code/src')
sys.path.insert(0, '/home/bendm/Thesis/project/code/src/models/tuss/base')

import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from models.tuss.inference import TUSSInference

# Load demo audio
demo_path = '/home/bendm/Thesis/project/code/src/validation_functions/demo_output/520539__szegvari__forest-wind-birds-tree-airplane-mastered.wav'
data, sr_orig = sf.read(demo_path, always_2d=True)
waveform_orig = torch.from_numpy(data.T).float()

print("="*70)
print("FULL INFERENCE PIPELINE ANALYSIS")
print("="*70)
print(f"\nOriginal demo audio: {waveform_orig.shape}, sr={sr_orig} Hz")
print(f"Duration: {waveform_orig.shape[-1] / sr_orig:.2f}s")
print(f"Max amplitude: {waveform_orig.abs().max():.4f}")

# Load the model
ckpt_path = "/home/bendm/Thesis/project/code/src/models/tuss/checkpoints/20260423_105141"
print(f"\nLoading model from: {ckpt_path}")

inferencer = TUSSInference.from_checkpoint(
    ckpt_path,
    device="cpu",  # Use CPU for easier debugging
    coi_prompt=["airplane", "birds"],
    bg_prompt="background",
)

print(f"\nModel expects: sr={inferencer.sample_rate} Hz")

# Manually trace through preprocessing
print("\n" + "="*70)
print("STEP 1: Resampling")
print("="*70)

# Convert to mono
if waveform_orig.shape[0] > 1:
    waveform = waveform_orig.mean(dim=0)
else:
    waveform = waveform_orig.squeeze(0)

# Resample to model's rate
from common.audio_utils import create_high_quality_resampler
if sr_orig != inferencer.sample_rate:
    resampler = create_high_quality_resampler(sr_orig, inferencer.sample_rate)
    waveform_resampled = resampler(waveform.unsqueeze(0)).squeeze(0)
    print(f"Resampled from {sr_orig} Hz to {inferencer.sample_rate} Hz")
    print(f"Shape: {waveform.shape} -> {waveform_resampled.shape}")
else:
    waveform_resampled = waveform

# Compute spectrograms
def compute_mel_spectrogram(waveform, sr, n_fft=2048, hop=512):
    """Compute mel spectrogram."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True
    )
    
    mag = torch.abs(stft)
    mag_db = 20 * torch.log10(mag + 1e-8)
    
    return mag_db

print("\n" + "="*70)
print("STEP 2: Normalization")
print("="*70)

# Compute std for normalization (matching inference code)
std = waveform_resampled.std() + 1e-8
waveform_normalized = waveform_resampled / std

print(f"Original std: {std:.4f}")
print(f"Normalized max: {waveform_normalized.abs().max():.4f}")
print(f"Normalized std: {waveform_normalized.std():.4f}")

print("\n" + "="*70)
print("STEP 3: Model Forward Pass")
print("="*70)

# Run through model
with torch.inference_mode():
    x = waveform_normalized.unsqueeze(0)  # (1, T)
    prompts = [inferencer.prompts_list]
    
    print(f"Input shape: {x.shape}")
    print(f"Prompts: {prompts}")
    
    output = inferencer.model(x, prompts)  # (1, n_sources, T)
    
    print(f"Output shape: {output.shape}")
    print(f"Output max: {output.abs().max():.4f}")
    print(f"Output mean: {output.abs().mean():.4f}")

# Denormalize
sources = output[0].cpu() * std  # (n_sources, T)

print("\n" + "="*70)
print("STEP 4: Output Energy Analysis")
print("="*70)

plane = sources[0].numpy()
birds = sources[1].numpy()
background = sources[2].numpy()

plane_energy = np.sum(plane**2)
birds_energy = np.sum(birds**2)
bg_energy = np.sum(background**2)
total_energy = plane_energy + birds_energy + bg_energy
input_energy = np.sum(waveform_resampled.numpy()**2)

print(f"\nEnergy distribution:")
print(f"  Input:      {input_energy:.4e} (100.00%)")
print(f"  Plane:      {plane_energy:.4e} ({100*plane_energy/input_energy:.4f}%)")
print(f"  Birds:      {birds_energy:.4e} ({100*birds_energy/input_energy:.4f}%)")
print(f"  Background: {bg_energy:.4e} ({100*bg_energy/input_energy:.4f}%)")
print(f"  Total out:  {total_energy:.4e} ({100*total_energy/input_energy:.2f}%)")

# Check if background is just the input
input_norm = waveform_resampled.numpy() / (np.sqrt(input_energy) + 1e-8)
bg_norm = background / (np.sqrt(bg_energy) + 1e-8)
correlation = np.sum(input_norm * bg_norm)

print(f"\nBackground-Input correlation: {correlation:.6f}")
if correlation > 0.99:
    print("  ❌ CRITICAL: Background is essentially the input!")

# Generate spectrograms
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

spec_input = compute_mel_spectrogram(waveform_resampled, inferencer.sample_rate)
spec_plane = compute_mel_spectrogram(torch.from_numpy(plane), inferencer.sample_rate)
spec_birds = compute_mel_spectrogram(torch.from_numpy(birds), inferencer.sample_rate)
spec_bg = compute_mel_spectrogram(torch.from_numpy(background), inferencer.sample_rate)

for i, (spec, title) in enumerate([
    (spec_input, f'Input Mixture (Resampled to {inferencer.sample_rate} Hz)'),
    (spec_plane, 'Separated: Airplane'),
    (spec_birds, 'Separated: Birds'),
    (spec_bg, 'Separated: Background')
]):
    im = axes[i].imshow(
        spec.numpy(),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest',
        vmin=-80,
        vmax=0
    )
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Frequency (kHz)')
    
    freq = np.linspace(0, inferencer.sample_rate/2, spec.shape[0])
    yticks = np.arange(0, spec.shape[0], spec.shape[0]//6)
    yticklabels = [f'{freq[int(y)]/1000:.1f}' for y in yticks]
    axes[i].set_yticks(yticks)
    axes[i].set_yticklabels(yticklabels)
    
    plt.colorbar(im, ax=axes[i], label='Magnitude (dB)')

axes[-1].set_xlabel('Time (frames)')

plt.tight_layout()
output_path = '/home/bendm/Thesis/project/code/test_separation_outputs/full_inference_spectrograms.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved full inference spectrograms to: {output_path}")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print("1. Resampling appears to work correctly")
print("2. Check spectrograms to see if model is actually separating")
print("3. Energy distribution shows if separation is happening")
print("="*70)
