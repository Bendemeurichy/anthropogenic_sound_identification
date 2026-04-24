"""Check resampling quality and visualize spectrograms."""
import sys
sys.path.insert(0, '/home/bendm/Thesis/project/code/src')

import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from common.audio_utils import create_high_quality_resampler

# Load test input
wav_path = '/home/bendm/Thesis/project/code/test_separation_outputs/test_input.wav'
data, sr_orig = sf.read(wav_path, always_2d=True)
# data is (frames, channels), convert to (channels, frames)
waveform = torch.from_numpy(data.T).float()

print("="*70)
print("RESAMPLING ANALYSIS")
print("="*70)
print(f"\nOriginal audio: {waveform.shape}, sr={sr_orig} Hz")

# Convert to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
else:
    # Ensure it's (1, T) shape
    waveform = waveform.reshape(1, -1)

print(f"Duration: {waveform.shape[-1] / sr_orig:.2f}s")
print(f"Max amplitude: {waveform.abs().max():.4f}")

# Resample to model's expected rate (48 kHz)
target_sr = 48000
print(f"\nResampling from {sr_orig} Hz to {target_sr} Hz...")

resampler = create_high_quality_resampler(sr_orig, target_sr)
waveform_resampled = resampler(waveform)

print(f"Resampled audio: {waveform_resampled.shape}, sr={target_sr} Hz")
print(f"Duration: {waveform_resampled.shape[-1] / target_sr:.2f}s")
print(f"Max amplitude: {waveform_resampled.abs().max():.4f}")

# Calculate spectrograms
def compute_spectrogram(waveform, sr, nfft=2048, hop=512):
    """Compute magnitude spectrogram."""
    import torch.nn.functional as F
    
    # Add batch dimension if needed
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Compute STFT
    window = torch.hann_window(nfft)
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=window,
        center=True,
        return_complex=True
    )
    
    # Magnitude spectrogram
    mag = torch.abs(stft)
    # Convert to dB
    mag_db = 20 * torch.log10(mag + 1e-8)
    
    return mag_db

print("\nComputing spectrograms...")
spec_orig = compute_spectrogram(waveform.squeeze(0), sr_orig)
spec_resampled = compute_spectrogram(waveform_resampled.squeeze(0), target_sr)

print(f"Original spectrogram shape: {spec_orig.shape}")
print(f"Resampled spectrogram shape: {spec_resampled.shape}")

# Create figure with spectrograms
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Original
im1 = axes[0].imshow(
    spec_orig.numpy(),
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='nearest',
    vmin=-80,
    vmax=0
)
axes[0].set_title(f'Original Audio ({sr_orig} Hz)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time (frames)')
axes[0].set_ylabel('Frequency Bin')
axes[0].set_ylim([0, spec_orig.shape[0]])
plt.colorbar(im1, ax=axes[0], label='Magnitude (dB)')

# Add frequency axis labels
freq_orig = np.linspace(0, sr_orig/2, spec_orig.shape[0])
yticks = np.arange(0, spec_orig.shape[0], spec_orig.shape[0]//8)
yticklabels = [f'{freq_orig[int(y)]/1000:.1f}' for y in yticks]
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(yticklabels)
axes[0].set_ylabel('Frequency (kHz)')

# Resampled
im2 = axes[1].imshow(
    spec_resampled.numpy(),
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='nearest',
    vmin=-80,
    vmax=0
)
axes[1].set_title(f'Resampled Audio ({target_sr} Hz) - Kaiser Window Sinc Interpolation', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time (frames)')
axes[1].set_ylabel('Frequency Bin')
axes[1].set_ylim([0, spec_resampled.shape[0]])
plt.colorbar(im2, ax=axes[1], label='Magnitude (dB)')

# Add frequency axis labels
freq_resampled = np.linspace(0, target_sr/2, spec_resampled.shape[0])
yticks = np.arange(0, spec_resampled.shape[0], spec_resampled.shape[0]//8)
yticklabels = [f'{freq_resampled[int(y)]/1000:.1f}' for y in yticks]
axes[1].set_yticks(yticks)
axes[1].set_yticklabels(yticklabels)
axes[1].set_ylabel('Frequency (kHz)')

plt.tight_layout()
output_path = '/home/bendm/Thesis/project/code/test_separation_outputs/resampling_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved spectrogram comparison to: {output_path}")

# Check for aliasing or artifacts
print("\n" + "="*70)
print("ARTIFACT ANALYSIS:")
print("="*70)

# Check energy distribution
orig_energy = (waveform**2).sum().item()
resampled_energy = (waveform_resampled**2).sum().item()
energy_ratio = resampled_energy / orig_energy

print(f"\nEnergy preservation:")
print(f"  Original:   {orig_energy:.4e}")
print(f"  Resampled:  {resampled_energy:.4e}")
print(f"  Ratio:      {energy_ratio:.4f} ({'✓ Good' if 0.95 < energy_ratio < 1.05 else '❌ Energy changed significantly'})")

# Check for high-frequency content above Nyquist of original
nyquist_orig = sr_orig / 2
freq_bins_above_nyquist = spec_resampled.shape[0] * (nyquist_orig / (target_sr/2))
energy_above_nyquist = spec_resampled[int(freq_bins_above_nyquist):, :].pow(2).mean().item()
energy_below_nyquist = spec_resampled[:int(freq_bins_above_nyquist), :].pow(2).mean().item()

print(f"\nFrequency content above original Nyquist ({nyquist_orig/1000:.1f} kHz):")
print(f"  Energy below Nyquist: {energy_below_nyquist:.4e}")
print(f"  Energy above Nyquist: {energy_above_nyquist:.4e}")
print(f"  Ratio: {energy_above_nyquist / energy_below_nyquist:.6f}")
if energy_above_nyquist / energy_below_nyquist < 0.01:
    print("  ✓ Good: Minimal energy above original Nyquist (no significant aliasing)")
else:
    print("  ⚠ Warning: Some energy above original Nyquist")

# Save resampled audio for listening
output_wav = '/home/bendm/Thesis/project/code/test_separation_outputs/test_input_resampled_48k.wav'
sf.write(output_wav, waveform_resampled.squeeze(0).numpy(), target_sr)
print(f"\nSaved resampled audio to: {output_wav}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("Check the spectrogram comparison image to verify:")
print("1. No visible artifacts or discontinuities")
print("2. Frequency content below original Nyquist is preserved")
print("3. No significant aliasing in the resampled version")
print("="*70)
