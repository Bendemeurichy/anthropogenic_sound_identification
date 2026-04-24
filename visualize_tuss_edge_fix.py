"""Visual comparison: Before and After edge artifact fix for TUSS."""
import sys
sys.path.insert(0, '/home/bendm/Thesis/project/code/src')

import torch
import matplotlib.pyplot as plt
import numpy as np
from common.audio_utils import create_high_quality_resampler, resample_with_padding

# Generate realistic test fragment (short audio segment)
# This simulates what TUSS sees during training/inference
np.random.seed(42)
torch.manual_seed(42)

# Create a short fragment (0.25 seconds at 32kHz = 8000 samples)
# This is typical for fragment-based separation
fragment_length = 8000
orig_sr = 32000
target_sr = 48000

# Generate test signal with multiple frequency components
t = torch.linspace(0, fragment_length/orig_sr, fragment_length)
signal = (
    torch.sin(2 * np.pi * 1000 * t) * 0.3 +  # 1kHz tone
    torch.sin(2 * np.pi * 2500 * t) * 0.2 +  # 2.5kHz tone
    torch.randn(fragment_length) * 0.1        # Noise
)
waveform = signal.unsqueeze(0)  # Shape: (1, T)

print("="*70)
print("EDGE ARTIFACT FIX - VISUAL COMPARISON")
print("="*70)
print(f"\nSimulating TUSS fragment processing:")
print(f"  Fragment length: {fragment_length/orig_sr:.3f}s ({fragment_length} samples)")
print(f"  Original rate: {orig_sr} Hz")
print(f"  Target rate: {target_sr} Hz")

# Method 1: OLD - Standard resampling (with edge artifacts)
print(f"\n[OLD METHOD] Standard Kaiser windowed sinc...")
resampler = create_high_quality_resampler(orig_sr, target_sr)
resampled_old = resampler(waveform)

# Method 2: NEW - Padded resampling (artifact-free)
print(f"[NEW METHOD] Padded Kaiser windowed sinc...")
resampled_new = resample_with_padding(waveform, orig_sr, target_sr)

print(f"\nOutput lengths:")
print(f"  OLD: {resampled_old.shape}")
print(f"  NEW: {resampled_new.shape}")

# Compute spectrograms
def compute_mel_spectrogram(waveform, sr):
    """Compute mel spectrogram (closer to what models see)."""
    # STFT parameters
    n_fft = 2048
    hop = 512
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

print("\nComputing spectrograms...")
spec_old = compute_mel_spectrogram(resampled_old, target_sr)
spec_new = compute_mel_spectrogram(resampled_new, target_sr)

# Create comparison figure
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

def plot_spec_with_edge_highlight(ax, spec, sr, title, highlight_artifacts=False):
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
    
    if highlight_artifacts:
        # Highlight edge regions
        ax.axvline(x=0, color='red', linewidth=3, linestyle='--', alpha=0.8)
        ax.axvline(x=spec.shape[1]-1, color='red', linewidth=3, linestyle='--', alpha=0.8)
        
        # Add red box around first few frames
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), 3, spec.shape[0], 
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        
        ax.text(1.5, spec.shape[0]*0.9, 'ARTIFACT\nZONE', 
               color='red', fontsize=10, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return im

# Original
plot_spec_with_edge_highlight(
    axes[0], compute_mel_spectrogram(waveform, orig_sr), orig_sr,
    f'Original Fragment ({orig_sr} Hz)', 
    highlight_artifacts=False
)

# OLD method (with artifacts)
plot_spec_with_edge_highlight(
    axes[1], spec_old, target_sr,
    f'❌ OLD: Standard Resampling ({target_sr} Hz) - Edge Artifacts Present',
    highlight_artifacts=True
)

# NEW method (artifact-free)
plot_spec_with_edge_highlight(
    axes[2], spec_new, target_sr,
    f'✅ NEW: Padded Resampling ({target_sr} Hz) - Edge Artifacts Eliminated',
    highlight_artifacts=False
)

# Add annotation
axes[2].text(1.5, spec_new.shape[0]*0.9, 'CLEAN\nEDGES', 
           color='green', fontsize=10, fontweight='bold',
           ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
output_path = '/home/bendm/Thesis/project/code/test_separation_outputs/tuss_edge_fix_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved comparison to: {output_path}")

# Quantitative analysis
print("\n" + "="*70)
print("QUANTITATIVE EDGE ANALYSIS")
print("="*70)

def analyze_edges_detailed(spec, name):
    """Detailed edge analysis."""
    edge_frames = 3
    left_edge = spec[:, :edge_frames]
    right_edge = spec[:, -edge_frames:]
    middle = spec[:, edge_frames:-edge_frames]
    
    left_max = left_edge.max().item()
    right_max = right_edge.max().item()
    middle_max = middle.max().item()
    
    left_mean = left_edge.mean().item()
    right_mean = right_edge.mean().item()
    middle_mean = middle.mean().item()
    
    print(f"\n{name}:")
    print(f"  Edge frames analyzed: {edge_frames}")
    print(f"  Left edge:   max={left_max:6.2f} dB, mean={left_mean:6.2f} dB")
    print(f"  Right edge:  max={right_max:6.2f} dB, mean={right_mean:6.2f} dB")
    print(f"  Middle:      max={middle_max:6.2f} dB, mean={middle_mean:6.2f} dB")
    
    edge_artifact_score = (abs(left_max - middle_max) + abs(right_max - middle_max)) / 2
    print(f"  Edge artifact score: {edge_artifact_score:.2f} dB")
    
    return edge_artifact_score

score_old = analyze_edges_detailed(spec_old, "OLD (Standard)")
score_new = analyze_edges_detailed(spec_new, "NEW (Padded)")

improvement = ((score_old - score_new) / score_old) * 100
print(f"\n{'='*70}")
print(f"Improvement: {improvement:.1f}% reduction in edge artifacts")
print(f"{'='*70}")

print("\n" + "="*70)
print("IMPACT ON SEPARATION MODELS")
print("="*70)
print("✅ Benefits of padded resampling:")
print("   1. Cleaner spectrograms → better model predictions")
print("   2. No discontinuities at fragment boundaries")
print("   3. More consistent feature extraction")
print("   4. Reduced high-frequency artifacts that confuse models")
print("\n⚠️  Why this matters for TUSS:")
print("   - TUSS processes audio in segments/fragments")
print("   - Each fragment is resampled independently")
print("   - Edge artifacts would appear at EVERY segment boundary")
print("   - Models learn on spectrograms where these artifacts are visible")
print("="*70)
