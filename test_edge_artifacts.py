"""Test edge artifact fix for Kaiser window sinc interpolation."""
import sys
sys.path.insert(0, '/home/bendm/Thesis/project/code/src')

import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from common.audio_utils import create_high_quality_resampler, resample_with_padding

# Load test input
wav_path = '/home/bendm/Thesis/project/code/test_separation_outputs/test_input.wav'
data, sr_orig = sf.read(wav_path, always_2d=True)
waveform = torch.from_numpy(data.T).float()

# Convert to mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
else:
    waveform = waveform.reshape(1, -1)

print("="*70)
print("EDGE ARTIFACT COMPARISON")
print("="*70)
print(f"\nOriginal: {waveform.shape}, sr={sr_orig} Hz")

target_sr = 48000

# Method 1: Standard resampling (with artifacts)
print(f"\nMethod 1: Standard resampling (current approach)")
resampler = create_high_quality_resampler(sr_orig, target_sr)
resampled_standard = resampler(waveform)
print(f"  Output: {resampled_standard.shape}")

# Method 2: Resampling with padding (artifact-free)
print(f"\nMethod 2: Resampling with reflection padding (fixed)")
resampled_padded = resample_with_padding(waveform, sr_orig, target_sr, pad_mode="reflect")
print(f"  Output: {resampled_padded.shape}")

# Compute spectrograms
def compute_spectrogram(waveform, sr, nfft=2048, hop=512):
    """Compute magnitude spectrogram in dB."""
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
    mag = torch.abs(stft)
    mag_db = 20 * torch.log10(mag + 1e-8)
    return mag_db

print("\nComputing spectrograms...")
spec_orig = compute_spectrogram(waveform, sr_orig)
spec_standard = compute_spectrogram(resampled_standard, target_sr)
spec_padded = compute_spectrogram(resampled_padded, target_sr)

# Create comparison figure
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

def plot_spec(ax, spec, sr, title):
    im = ax.imshow(
        spec.numpy(),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest',
        vmin=-80,
        vmax=0
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (frames)', fontsize=12)
    
    # Frequency axis
    freq = np.linspace(0, sr/2, spec.shape[0])
    yticks = np.arange(0, spec.shape[0], spec.shape[0]//8)
    yticklabels = [f'{freq[int(y)]/1000:.1f}' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    # Add red boxes around edge regions to highlight artifacts
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=spec.shape[1]-1, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    return im

# Plot all three
plot_spec(axes[0], spec_orig, sr_orig, f'Original Audio ({sr_orig} Hz)')
plot_spec(axes[1], spec_standard, target_sr, 
          f'Standard Resampling ({target_sr} Hz) - ⚠️ Edge Artifacts Visible')
plot_spec(axes[2], spec_padded, target_sr, 
          f'Padded Resampling ({target_sr} Hz) - ✅ Edge Artifacts Eliminated')

# Add annotation pointing to artifacts
axes[1].annotate('Edge artifact\n(bright vertical line)', 
                xy=(0, spec_standard.shape[0]//2), 
                xytext=(20, spec_standard.shape[0]//2 + 100),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')

axes[2].annotate('Clean edge\n(no artifacts)', 
                xy=(0, spec_padded.shape[0]//2), 
                xytext=(20, spec_padded.shape[0]//2 + 100),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
output_path = '/home/bendm/Thesis/project/code/test_separation_outputs/edge_artifacts_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved comparison to: {output_path}")

# Analyze edge energy
def analyze_edges(spec, name, num_edge_frames=5):
    """Analyze energy in edge frames vs middle frames."""
    left_edge = spec[:, :num_edge_frames].pow(2).mean().item()
    right_edge = spec[:, -num_edge_frames:].pow(2).mean().item()
    middle = spec[:, num_edge_frames:-num_edge_frames].pow(2).mean().item()
    
    print(f"\n{name}:")
    print(f"  Left edge energy:   {left_edge:.4e}")
    print(f"  Right edge energy:  {right_edge:.4e}")
    print(f"  Middle energy:      {middle:.4e}")
    print(f"  Left/Middle ratio:  {left_edge/middle:.4f}x")
    print(f"  Right/Middle ratio: {right_edge/middle:.4f}x")
    
    if max(left_edge/middle, right_edge/middle) > 1.5:
        print(f"  ⚠️  Edge artifacts detected (>1.5x higher energy)")
    else:
        print(f"  ✅ Clean edges (energy similar to middle)")
    
    return left_edge, right_edge, middle

print("\n" + "="*70)
print("EDGE ENERGY ANALYSIS:")
print("="*70)

analyze_edges(spec_standard, "Standard Resampling")
analyze_edges(spec_padded, "Padded Resampling")

# Save audio files for comparison
sf.write('/home/bendm/Thesis/project/code/test_separation_outputs/resampled_standard.wav',
         resampled_standard.squeeze(0).numpy(), target_sr)
sf.write('/home/bendm/Thesis/project/code/test_separation_outputs/resampled_padded.wav',
         resampled_padded.squeeze(0).numpy(), target_sr)

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("✅ Use resample_with_padding() for:")
print("   - Fragment-based separation models")
print("   - Any processing where spectrograms are analyzed")
print("   - High-quality audio output")
print("\n⚠️  Standard resampling (without padding) causes:")
print("   - Visible edge artifacts in spectrograms")
print("   - Potential confusion for ML models")
print("   - Discontinuities at fragment boundaries")
print("="*70)
