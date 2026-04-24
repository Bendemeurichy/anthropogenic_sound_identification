"""Visualize spectrograms of separation outputs without loading the model."""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch

def compute_spectrogram(waveform, sr, n_fft=2048, hop=512):
    """Compute magnitude spectrogram."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    
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
    
    return mag_db.numpy()

output_dir = '/home/bendm/Thesis/project/code/src/validation_functions/demo_output'

# Load all files
print("Loading audio files...")
mixture, sr = sf.read(f'{output_dir}/mixture.wav')
plane, _ = sf.read(f'{output_dir}/separated_plane.wav')
bird, _ = sf.read(f'{output_dir}/separated_bird.wav')
bg, _ = sf.read(f'{output_dir}/separated_background.wav')

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(mixture) / sr:.2f}s")

# Compute spectrograms
print("\nComputing spectrograms...")
spec_mix = compute_spectrogram(mixture, sr)
spec_plane = compute_spectrogram(plane, sr)
spec_bird = compute_spectrogram(bird, sr)
spec_bg = compute_spectrogram(bg, sr)

print(f"Spectrogram shapes: {spec_mix.shape}")

# Create comprehensive visualization
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

specs = [
    (spec_mix, 'Input Mixture', 'viridis'),
    (spec_plane, 'Separated: Airplane', 'viridis'),
    (spec_bird, 'Separated: Birds', 'viridis'),
    (spec_bg, 'Separated: Background', 'viridis')
]

for i, (spec, title, cmap) in enumerate(specs):
    im = axes[i].imshow(
        spec,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
        vmin=-80,
        vmax=0
    )
    axes[i].set_title(title, fontsize=14, fontweight='bold')
    
    # Add frequency axis
    freq = np.linspace(0, sr/2, spec.shape[0])
    yticks = np.arange(0, spec.shape[0], spec.shape[0]//8)
    yticklabels = [f'{freq[int(y)]/1000:.1f}' for y in yticks]
    axes[i].set_yticks(yticks)
    axes[i].set_yticklabels(yticklabels)
    axes[i].set_ylabel('Frequency (kHz)')
    
    # Add time axis
    time = np.linspace(0, len(mixture)/sr, spec.shape[1])
    xticks = np.linspace(0, spec.shape[1]-1, 6)
    xticklabels = [f'{time[int(x)]:.1f}' for x in xticks]
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels(xticklabels)
    
    plt.colorbar(im, ax=axes[i], label='Magnitude (dB)')

axes[-1].set_xlabel('Time (s)')

plt.tight_layout()
output_path = '/home/bendm/Thesis/project/code/test_separation_outputs/demo_separation_spectrograms.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved spectrograms to: {output_path}")

# Create difference spectrograms
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10))

# Compute differences (how much each output differs from mixture)
diff_plane = spec_plane - spec_mix
diff_bird = spec_bird - spec_mix
diff_bg = spec_bg - spec_mix

diffs = [
    (diff_plane, 'Airplane - Mixture (dB difference)', 'RdBu_r'),
    (diff_bird, 'Birds - Mixture (dB difference)', 'RdBu_r'),
    (diff_bg, 'Background - Mixture (dB difference)', 'RdBu_r')
]

for i, (diff, title, cmap) in enumerate(diffs):
    im = axes2[i].imshow(
        diff,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
        vmin=-40,
        vmax=40
    )
    axes2[i].set_title(title, fontsize=14, fontweight='bold')
    
    # Add frequency axis
    freq = np.linspace(0, sr/2, diff.shape[0])
    yticks = np.arange(0, diff.shape[0], diff.shape[0]//8)
    yticklabels = [f'{freq[int(y)]/1000:.1f}' for y in yticks]
    axes2[i].set_yticks(yticks)
    axes2[i].set_yticklabels(yticklabels)
    axes2[i].set_ylabel('Frequency (kHz)')
    
    # Add time axis
    time = np.linspace(0, len(mixture)/sr, diff.shape[1])
    xticks = np.linspace(0, diff.shape[1]-1, 6)
    xticklabels = [f'{time[int(x)]:.1f}' for x in xticks]
    axes2[i].set_xticks(xticks)
    axes2[i].set_xticklabels(xticklabels)
    
    plt.colorbar(im, ax=axes2[i], label='dB difference (0 = same as mixture)')

axes2[-1].set_xlabel('Time (s)')

plt.tight_layout()
output_path2 = '/home/bendm/Thesis/project/code/test_separation_outputs/demo_separation_differences.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved difference spectrograms to: {output_path2}")

print("\n" + "="*70)
print("INTERPRETATION GUIDE:")
print("="*70)
print("\nFor absolute spectrograms:")
print("  - If plane/bird outputs are mostly dark (-80 dB), they're nearly silent")
print("  - If background looks identical to mixture, model isn't separating")
print("\nFor difference spectrograms:")
print("  - White/red areas: output is LOUDER than mixture (impossible, indicates problem)")
print("  - Blue areas: output is QUIETER than mixture")
print("  - Near-zero (white): output matches mixture exactly")
print("  - If background difference is near-zero everywhere, it's copying the input")
print("="*70)
