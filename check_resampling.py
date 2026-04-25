"""Check resampling quality and visualize spectrograms."""

import sys

sys.path.insert(0, "/home/bendm/Thesis/project/code/src")

import torch
import soundfile as sf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common.audio_utils import resample_with_padding

# Load test input
wav_path = "/home/bendm/Thesis/project/code/test_separation_outputs/test_input.wav"
data, sr_orig = sf.read(wav_path, always_2d=True)
# data is (frames, channels), convert to (channels, frames)
waveform = torch.from_numpy(data.T).float()

print("=" * 70)
print("RESAMPLING ANALYSIS")
print("=" * 70)
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

waveform_resampled = resample_with_padding(waveform, sr_orig, target_sr)

print(f"Resampled audio: {waveform_resampled.shape}, sr={target_sr} Hz")
print(f"Duration: {waveform_resampled.shape[-1] / target_sr:.2f}s")
print(f"Max amplitude: {waveform_resampled.abs().max():.4f}")


# Calculate spectrograms
def compute_spectrogram(waveform, sr, nfft=2048, hop=512):
    """Compute magnitude spectrogram."""

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
        return_complex=True,
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
time_orig = np.linspace(0, waveform.shape[-1] / sr_orig, spec_orig.shape[1])
time_resampled = np.linspace(
    0, waveform_resampled.shape[-1] / target_sr, spec_resampled.shape[1]
)
freq_orig = np.linspace(0, sr_orig / 2 / 1000, spec_orig.shape[0])
freq_resampled = np.linspace(0, target_sr / 2 / 1000, spec_resampled.shape[0])

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=False,
    vertical_spacing=0.12,
    subplot_titles=[
        f"<b>Original ({sr_orig / 1000:.0f} kHz)</b>",
        f"<b>Resampled ({target_sr / 1000:.0f} kHz) — Kaiser window sinc interpolation</b>",
    ],
)

colorscale = "Magma"
zmin, zmax = -80, 0

fig.add_trace(
    go.Heatmap(
        z=spec_orig.numpy(),
        x=time_orig,
        y=freq_orig,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=dict(text="Magnitude (dB)", side="right"),
            thickness=15,
            len=1.0,
            y=0.5,
            yanchor="middle",
        ),
        showscale=True,
        name="Original",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Heatmap(
        z=spec_resampled.numpy(),
        x=time_resampled,
        y=freq_resampled,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        showscale=False,
        name="Resampled",
    ),
    row=2,
    col=1,
)

axis_title_font = dict(family="Arial, sans-serif", size=13, color="black")
fig.update_xaxes(title_text="<b>Time (s)</b>", title_font=axis_title_font, row=1, col=1)
fig.update_xaxes(title_text="<b>Time (s)</b>", title_font=axis_title_font, row=2, col=1)
fig.update_yaxes(
    title_text="<b>Frequency (kHz)</b>", title_font=axis_title_font, row=1, col=1
)
fig.update_yaxes(
    title_text="<b>Frequency (kHz)</b>", title_font=axis_title_font, row=2, col=1
)

fig.update_layout(
    font=dict(family="Arial, sans-serif", size=13),
    width=900,
    height=750,
    margin=dict(l=70, r=100, t=60, b=60),
)

output_path = (
    "/home/bendm/Thesis/project/code/test_separation_outputs/resampling_comparison.pdf"
)
png_path = output_path.replace(".pdf", ".png")
fig.write_image(output_path)
fig.write_image(png_path, scale=2)
print(f"\nSaved spectrogram comparison to: {output_path}")
# Check for aliasing or artifacts
print("\n" + "=" * 70)
print("ARTIFACT ANALYSIS:")
print("=" * 70)

# Check energy distribution
orig_energy = (waveform**2).sum().item()
resampled_energy = (waveform_resampled**2).sum().item()
energy_ratio = resampled_energy / orig_energy

print("\nEnergy preservation:")
print(f"  Original:   {orig_energy:.4e}")
print(f"  Resampled:  {resampled_energy:.4e}")
print(
    f"  Ratio:      {energy_ratio:.4f} ({'✓ Good' if 0.95 < energy_ratio < 1.05 else '❌ Energy changed significantly'})"
)

# Check for high-frequency content above Nyquist of original
nyquist_orig = sr_orig / 2
freq_bins_above_nyquist = spec_resampled.shape[0] * (nyquist_orig / (target_sr / 2))
energy_above_nyquist = (
    spec_resampled[int(freq_bins_above_nyquist) :, :].pow(2).mean().item()
)
energy_below_nyquist = (
    spec_resampled[: int(freq_bins_above_nyquist), :].pow(2).mean().item()
)

print(f"\nFrequency content above original Nyquist ({nyquist_orig/1000:.1f} kHz):")
print(f"  Energy below Nyquist: {energy_below_nyquist:.4e}")
print(f"  Energy above Nyquist: {energy_above_nyquist:.4e}")
print(f"  Ratio: {energy_above_nyquist / energy_below_nyquist:.6f}")
if energy_above_nyquist / energy_below_nyquist < 0.01:
    print("  ✓ Good: Minimal energy above original Nyquist (no significant aliasing)")
else:
    print("  ⚠ Warning: Some energy above original Nyquist")

# Save resampled audio for listening
output_wav = "/home/bendm/Thesis/project/code/test_separation_outputs/test_input_resampled_48k.wav"
sf.write(output_wav, waveform_resampled.squeeze(0).numpy(), target_sr)
print(f"\nSaved resampled audio to: {output_wav}")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("Check the spectrogram comparison image to verify:")
print("1. No visible artifacts or discontinuities")
print("2. Frequency content below original Nyquist is preserved")
print("3. No significant aliasing in the resampled version")
print("=" * 70)
