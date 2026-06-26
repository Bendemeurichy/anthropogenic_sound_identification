"""Simple TUSS spectrogram comparison with and without Gaussian white noise."""
import argparse
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.tuss.inference import TUSSInference  # noqa: E402


def compute_spectrogram(waveform, sr, n_fft=2048, hop=512):
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )

    mag = torch.abs(stft)
    mag_db = 20 * torch.log10(mag + 1e-8)
    mag_db = mag_db + 80
    return mag_db.numpy()


def add_gaussian_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.standard_normal(signal.shape) * np.sqrt(noise_power)
    return signal + noise


def load_mono(path):
    signal, sr = sf.read(path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal, sr


def add_heatmap(fig, spec, time_axis, freq_axis, row, col,
                vmin, vmax, showscale=False):
    fig.add_trace(
        go.Heatmap(
            z=spec,
            x=time_axis,
            y=freq_axis,
            colorscale="Magma",
            zmin=vmin,
            zmax=vmax,
            showscale=showscale,
            colorbar=dict(
                title="Level",
                thickness=12,
                len=0.25,
                x=1.01,
                y=0.12,
            ) if showscale else None,
            hovertemplate=(
                "Time: %{x:.2f}s<br>Freq: %{y:.0f}Hz<br>%{z:.1f}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare clean/noisy input spectrograms and their TUSS outputs."
    )
    parser.add_argument("wav_path", type=str, help="Path to input WAV file")
    parser.add_argument(
        "--checkpoint", type=str,
        default="src/models/tuss/checkpoints/multi_coi_14_05",
        help="Path to TUSS checkpoint directory"
    )
    parser.add_argument(
        "--coi-prompt", type=str, default="airplane",
        help="COI prompt for TUSS (default: airplane)"
    )
    parser.add_argument(
        "--bg-prompt", type=str, default="background",
        help="Background prompt for TUSS (default: background)"
    )
    parser.add_argument(
        "--snr", type=float, default=45.0,
        help="SNR for Gaussian white noise in dB (default: 45, mild visible noise)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the output HTML"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible noise (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run TUSS on (default: auto-detect CUDA, fallback to CPU)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=6.0,
        help="Max audio duration in seconds (truncated at end; default: 6.0, matches TUSS segment)"
    )
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(args.seed)

    print(f"Loading: {args.wav_path}")
    signal, sr = load_mono(args.wav_path)

    print(f"Loading TUSS from: {args.checkpoint}")
    tuss = TUSSInference.from_checkpoint(
        args.checkpoint,
        device=device,
        coi_prompt=args.coi_prompt,
        bg_prompt=args.bg_prompt,
    )
    model_sr = tuss.sample_rate
    print(f"  TUSS sample rate: {model_sr} Hz")

    if sr != model_sr:
        print(f"  Resampling input from {sr} Hz to {model_sr} Hz...")
        import torchaudio
        signal_t = torch.from_numpy(signal).float().unsqueeze(0)
        signal_t = torchaudio.functional.resample(signal_t, sr, model_sr)
        signal = signal_t.squeeze(0).numpy()
        sr = model_sr

    duration = len(signal) / sr
    if args.max_duration and duration > args.max_duration:
        max_samples = int(args.max_duration * sr)
        signal = signal[:max_samples]
        duration = args.max_duration
        print(f"  Truncated to {duration:.2f}s ({max_samples} samples)")

    print(f"  Sample rate: {sr} Hz, Duration: {duration:.2f}s")

    print(f"Running TUSS separation on clean mixture...")
    with torch.inference_mode():
        sources_clean = tuss.separate_waveform(
            torch.from_numpy(signal).float().to(device)
        )
    coi_clean = tuss.get_coi_by_name(sources_clean, args.coi_prompt).cpu()

    print(f"Adding Gaussian white noise at SNR={args.snr} dB (seed={args.seed})...")
    noisy_signal = add_gaussian_noise(signal, args.snr, rng)

    print(f"Running TUSS separation on noisy mixture...")
    with torch.inference_mode():
        sources_noisy = tuss.separate_waveform(
            torch.from_numpy(noisy_signal).float().to(device)
        )
    coi_noisy = tuss.get_coi_by_name(sources_noisy, args.coi_prompt).cpu()

    min_len = min(len(signal), len(coi_clean), len(coi_noisy))
    signal = signal[:min_len]
    noisy_signal = noisy_signal[:min_len]
    coi_clean = coi_clean[:min_len]
    coi_noisy = coi_noisy[:min_len]

    print("Computing spectrograms...")
    spec_clean = compute_spectrogram(torch.from_numpy(signal).float(), sr)
    spec_noisy = compute_spectrogram(torch.from_numpy(noisy_signal).float(), sr)
    spec_coi_clean = compute_spectrogram(coi_clean, sr)
    spec_coi_noisy = compute_spectrogram(coi_noisy, sr)

    freq_axis = np.linspace(0, sr / 2, spec_clean.shape[0])
    time_axis = np.linspace(0, len(signal) / sr, spec_clean.shape[1])

    rows, cols = 2, 2
    subplot_titles = [
        "Clean input",
        f"TUSS {args.coi_prompt} output from clean input",
        f"Noisy input (Gaussian, SNR={args.snr:g} dB)",
        f"TUSS {args.coi_prompt} output from noisy input",
    ]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    mag_vmin = min(spec_clean.min(), spec_noisy.min(), spec_coi_clean.min(), spec_coi_noisy.min())
    mag_vmax = max(spec_clean.max(), spec_noisy.max(), spec_coi_clean.max(), spec_coi_noisy.max())

    panels = [
        (spec_clean,     1, 1, mag_vmin, mag_vmax, False),
        (spec_coi_clean, 1, 2, mag_vmin, mag_vmax, False),
        (spec_noisy,     2, 1, mag_vmin, mag_vmax, False),
        (spec_coi_noisy, 2, 2, mag_vmin, mag_vmax, True),
    ]

    for spec, row, col, vmin, vmax, show in panels:
        add_heatmap(fig, spec, time_axis, freq_axis, row, col, vmin, vmax, show)

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(title_text="Time (s)", row=r, col=c)
            if c == 1:
                fig.update_yaxes(title_text="Frequency (Hz)", row=r, col=c)

    fig.update_layout(
        title_text=f"Clean vs. Noisy TUSS Separation - {Path(args.wav_path).name}",
        height=900,
        width=1200,
        hovermode="closest",
        margin=dict(l=60, r=60, t=80, b=60),
    )

    input_path = Path(args.wav_path)
    name = input_path.stem
    output_path = args.output or str(input_path.parent / f"{name}_tuss_noise_comparison.html")
    fig.write_html(output_path)
    print(f"Saved interactive comparison to: {output_path}")


if __name__ == "__main__":
    main()
