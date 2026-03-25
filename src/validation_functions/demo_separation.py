"""Demo separation — spectrogram visualisation utilities.

Loads WAV files, computes Mel-spectrograms, and saves plots suitable for
inclusion in papers or reports.

Exposed utilities:

* ``load_wav`` – load a WAV file at its native (or a specified) sample rate,
  returning a ``(waveform, sample_rate)`` tuple.
* ``compute_spectrogram`` – compute a log-magnitude Mel-spectrogram.
* ``compute_energy_metrics`` – compute RMS and SEL in dBFS.
* ``plot_spectrogram`` – render a single spectrogram and save a PNG.
* ``plot_spectrogram_from_wav`` – convenience one-call wrapper.
* ``plot_multiple_wav_spectrograms`` – batch version of the above.
* ``plot_combined_spectrograms`` – vertically stacked comparison figure from
  pre-computed spectrograms.
* ``plot_combined_spectrograms_from_wavs`` – same, loading WAVs automatically.

All functions can be imported and used independently.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plots save without display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default sample rate — used as a fallback when no sr can be inferred from a
# file.  Callers that pass explicit sr= or use the auto-detecting helpers
# (load_wav with target_sr=None, plot_*_from_wav* with sr=None) will never
# rely on this value.
SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
# Human hearing range — used to cap the y-axis of spectrogram plots so that
# we never display empty frequency bins above the signal's Nyquist.
HUMAN_HEARING_MAX = 20_000
# Minimum sample rate that gives Nyquist > HUMAN_HEARING_MAX.
# 44 100 Hz (CD standard) gives Nyquist = 22 050 Hz, safely above 20 kHz.
# Use as an explicit display_sr override when you want to force all panels in
# a comparison figure onto a common 0–20 kHz axis (only meaningful when the
# source audio is already natively sampled at or above this rate).
DISPLAY_MIN_SR = 44_100


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def load_wav(path: Path, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """Load a WAV file and return a ``(waveform, sample_rate)`` tuple.

    If *target_sr* is given the waveform is resampled to that rate before
    being returned and *sample_rate* in the return value reflects the
    requested rate.  If *target_sr* is ``None`` (the default) the audio is
    loaded at its native sample rate with no resampling.

    Args:
        path:      Path to the WAV file.
        target_sr: Desired sample rate in Hz, or ``None`` to keep the native
                   rate.

    Returns:
        ``(wav, sr)`` where *wav* is a 1-D float32 tensor and *sr* is the
        sample rate actually used.
    """
    wav, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(wav)

    # Handle multi-channel: convert to (channels, samples) if needed
    if wav.ndim == 2:
        wav = wav.T  # (samples, channels) -> (channels, samples)
    else:
        wav = wav.unsqueeze(0)  # (samples,) -> (1, samples)

    if target_sr is not None and sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr

    # Mix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    return wav, sr


def compute_spectrogram(
    wav: torch.Tensor,
    sr: int = SAMPLE_RATE,
    display_sr: int | None = None,
    n_mels: int = 128,
) -> tuple[np.ndarray, int]:
    """
    Return a log-magnitude Mel-spectrogram (n_mels x time) as a numpy array.

    The spectrogram is computed at ``sr`` by default.  No upsampling is
    performed — the y-axis of the resulting figure will only extend to
    ``sr / 2`` (the true Nyquist of the signal).  Upsampling a 16 kHz signal
    to a higher rate does not add spectral content; it would only produce a
    misleading empty band above 8 kHz.

    If ``display_sr`` is provided *and* differs from ``sr`` the waveform is
    resampled to ``display_sr`` before the STFT.  Use this only when you
    deliberately want all panels in a multi-panel figure to share the same
    frequency axis (e.g. pass ``display_sr=DISPLAY_MIN_SR`` to guarantee
    coverage up to 20 kHz for signals already sampled at ≥ 44 100 Hz).

    Returns ``(mel_spec_db, used_sr)`` where ``used_sr`` is the sample rate
    actually used (useful when labelling axes).
    """
    # Only resample when explicitly requested
    if display_sr is not None and display_sr != sr:
        wav = torchaudio.transforms.Resample(sr, display_sr)(wav.unsqueeze(0)).squeeze(
            0
        )
        sr = display_sr

    # Compute Mel-spectrogram (torchaudio returns power by default)
    # MelSpectrogram expects input shape (channel, samples)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=n_mels,
        power=2.0,
    )
    mel_spec = mel_transform(wav.unsqueeze(0))[0]  # (n_mels, time)

    # Convert power to dBFS (reference: full scale amplitude of 1.0)
    # For a full-scale sinusoid, power = 0.5, so 10*log10(0.5) ≈ -3 dBFS
    # But for consistency with RMS (which uses amplitude), we reference to power=1.0
    # This gives us: dBFS = 10*log10(power / 1.0) = 10*log10(power)
    # Maximum possible power for a signal with amplitude 1.0 is 1.0, giving 0 dBFS
    mel_spec_db = 10.0 * torch.log10(mel_spec + 1e-8)

    return mel_spec_db.cpu().numpy(), sr


def compute_energy_metrics(wav: torch.Tensor, sr: int) -> dict:
    """Compute RMS and SEL for a waveform.

    Both values are expressed in dBFS (reference = full-scale, i.e. amplitude 1.0).

    * RMS (dBFS) = 20 · log10( sqrt( mean(x²) ) ) — independent of clip duration.
    * SEL (dBFS) = 10 · log10( sum(x²) / sr )  — Sound Exposure Level; normalised
      by sample rate to give units comparable to Pa²·s, but remains dependent on
      clip duration (longer clips accumulate more energy).

    Args:
        wav: 1-D waveform tensor.
        sr:  Sample rate in Hz (used to normalise SEL).

    Returns:
        dict with keys ``rms_db`` (float) and ``sel_db`` (float).
    """
    eps = 1e-12
    x = wav.detach().float()
    rms = x.pow(2).mean().sqrt()
    rms_db = 20.0 * float(torch.log10(rms + eps))
    sel = x.pow(2).sum() / sr
    sel_db = 10.0 * float(torch.log10(sel + eps))
    return {"rms_db": rms_db, "sel_db": sel_db}


def plot_spectrogram(
    spec: np.ndarray,
    title: str,
    save_path: Path,
    sr: int = SAMPLE_RATE,
    metrics: dict | None = None,
) -> None:
    """Plot a Mel-spectrogram (mel x time) with informative y-axis (Hz) and save to *save_path*.

    If *metrics* is provided (a dict with keys ``rms_db`` and ``sel_db`` as returned by
    :func:`compute_energy_metrics`) the values are rendered as a small annotation box in
    the upper-right corner of the axes.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    # time extent in seconds
    time_sec = spec.shape[1] * HOP_LENGTH / sr
    n_mels = spec.shape[0]

    # We display the matrix with vertical axis as Mel bands (0..n_mels)
    # Map a few Mel-band positions back to Hz for readable tick labels.
    # Uses the standard mel <-> hz conversion:
    #   mel = 2595 * log10(1 + f/700)
    #   f   = 700 * (10^(mel/2595) - 1)
    def mel_to_hz(mel_val: float) -> float:
        return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

    # Cap the displayed frequency range at the signal's Nyquist, or at the
    # human hearing limit (20 kHz) if the signal covers the full audible band.
    nyquist = sr / 2.0
    display_max_hz = min(HUMAN_HEARING_MAX, nyquist)

    # mel values for 0 Hz and the display ceiling
    mel_max = 2595.0 * np.log10(1.0 + nyquist / 700.0)
    mel_display_max = 2595.0 * np.log10(1.0 + display_max_hz / 700.0)

    # Fractional bin index corresponding to the display ceiling
    display_max_bin = (mel_display_max / mel_max) * (n_mels - 1)

    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0, time_sec, 0, n_mels),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    # Clip y-axis to the display ceiling
    ax.set_ylim(0, display_max_bin)

    # Choose a small set of ticks evenly spaced on the mel axis up to the ceiling
    num_ticks = 6
    mel_ticks = np.linspace(0.0, mel_display_max, num_ticks)
    # Convert mel values to corresponding bin indices in [0, n_mels-1]
    mel_bin_indices = (mel_ticks / mel_max) * (n_mels - 1)
    # Positions for imshow (extent uses 0..n_mels), so use those indices directly
    y_positions = mel_bin_indices

    # Format Hz labels (use k for thousands)
    def fmt_hz(hz: float) -> str:
        if hz >= 1000:
            return f"{hz / 1000:.1f}k"
        return f"{int(hz)}"

    hz_labels = [fmt_hz(mel_to_hz(m)) for m in mel_ticks]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(hz_labels)

    # Optional energy metrics annotation (upper-right corner)
    if metrics is not None:
        annotation = (
            f"RMS: {metrics['rms_db']:.1f} dBFS\nSEL: {metrics['sel_db']:.1f} dBFS"
        )
        ax.text(
            0.98,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Saved spectrogram -> {save_path}")


def plot_spectrogram_from_wav(
    wav_path: Path,
    save_path: Path,
    title: str | None = None,
    sr: int | None = None,
) -> None:
    """Load a WAV file, compute its spectrogram, and save a plot.

    Convenience wrapper that handles I/O and plotting in one call.

    If *sr* is ``None`` (the default) the audio is loaded at its native
    sample rate.  Pass an explicit *sr* to force resampling before plotting.
    The y-axis always extends to the Nyquist of the actual sample rate used —
    no artificial upsampling is ever performed.
    """
    wav, used_sr = load_wav(wav_path, target_sr=sr)
    spec, used_sr = compute_spectrogram(wav, sr=used_sr)
    if title is None:
        title = f"Spectrogram - {wav_path.name}"
    plot_spectrogram(spec, title, save_path, sr=used_sr)


def plot_multiple_wav_spectrograms(
    wav_paths: list[Path],
    out_dir: Path,
    sr: int | None = None,
) -> None:
    """Plot spectrograms for a list of WAV files and save them to *out_dir*.

    Each output file is named with a timestamp to avoid collisions.
    If *sr* is ``None`` each file is loaded at its own native sample rate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for wav_path in wav_paths:
        stem = wav_path.stem.replace(" ", "_")
        save_path = out_dir / f"spectrogram_{stem}_{ts}.png"
        plot_spectrogram_from_wav(wav_path, save_path, title=wav_path.name, sr=sr)


def plot_combined_spectrograms(
    specs: list[np.ndarray],
    titles: list[str],
    save_path: Path,
    sr: int = SAMPLE_RATE,
    metrics: list[dict] | None = None,
    ref_idx: int | None = None,
    delta_indices: list[int] | None = None,
) -> None:
    """Create a combined vertical figure from a list of Mel-spectrograms.

    Each input in ``specs`` is expected to be a Mel-spectrogram (n_mels x time).
    The y-axis tick labels are shown in Hz for easier human interpretation.

    Args:
        specs:    List of Mel-spectrograms (n_mels x time).
        titles:   Per-panel titles.
        save_path: Output PNG path.
        sr:       Sample rate used for axis labelling.
        metrics:  Optional list of dicts (one per panel) as returned by
                  :func:`compute_energy_metrics`, with keys ``rms_db`` and
                  ``sel_db``.  When supplied a narrow text panel is rendered
                  to the right of each spectrogram.
        ref_idx:  Index of the panel to treat as the energy reference.  When
                  provided, panels specified in delta_indices display ΔRMS and
                  ΔSEL relative to that panel.
        delta_indices: List of panel indices that should show delta values
                  relative to ref_idx. If None and ref_idx is provided, all
                  panels except ref_idx show deltas.
    """
    import matplotlib.gridspec as gridspec

    n_plots = len(specs)
    has_metrics = metrics is not None and len(metrics) == n_plots

    # Total figure width: slightly wider when the metrics column is present
    fig_w = 14 if has_metrics else 12
    fig = plt.figure(figsize=(fig_w, 3.5 * n_plots))

    if has_metrics:
        gs = gridspec.GridSpec(
            n_plots,
            2,
            width_ratios=[5, 1],
            figure=fig,
            hspace=0.45,
            wspace=0.05,
        )
    else:
        gs = gridspec.GridSpec(n_plots, 1, figure=fig, hspace=0.45)

    def mel_to_hz(mel_val: float) -> float:
        return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

    def fmt_hz(hz: float) -> str:
        return f"{hz / 1000:.1f}k" if hz >= 1000 else f"{int(hz)}"

    # Cap at the signal's Nyquist, or 20 kHz for audio that covers the full
    # audible band — never show empty frequency bins from artificial upsampling.
    nyquist = sr / 2.0
    display_max_hz = min(HUMAN_HEARING_MAX, nyquist)
    mel_max = 2595.0 * np.log10(1.0 + nyquist / 700.0)
    mel_display_max = 2595.0 * np.log10(1.0 + display_max_hz / 700.0)

    ref_metrics = metrics[ref_idx] if (has_metrics and ref_idx is not None) else None

    # If delta_indices not specified but ref_idx is, show deltas for all panels except ref_idx
    if delta_indices is None and ref_idx is not None:
        delta_indices = [i for i in range(n_plots) if i != ref_idx]

    # Compute global min/max dB values across all spectrograms for uniform scaling
    global_vmin = min(spec.min() for spec in specs)
    global_vmax = max(spec.max() for spec in specs)

    # Normalize spectrograms to relative scale (0 = min, max = dynamic range)
    normalized_specs = [spec - global_vmin for spec in specs]
    global_vmax_normalized = global_vmax - global_vmin

    for row, (spec, title) in enumerate(zip(normalized_specs, titles)):
        ax = fig.add_subplot(gs[row, 0])

        time_sec = spec.shape[1] * HOP_LENGTH / sr
        n_mels = spec.shape[0]
        display_max_bin = (mel_display_max / mel_max) * (n_mels - 1)

        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=(0, time_sec, 0, n_mels),
            vmin=0,
            vmax=global_vmax_normalized,
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(title)
        ax.set_ylim(0, display_max_bin)

        if row == n_plots - 1:
            ax.set_xlabel("Time (s)")

        num_ticks = 6
        mel_ticks = np.linspace(0.0, mel_display_max, num_ticks)
        mel_bin_indices = (mel_ticks / mel_max) * (n_mels - 1)
        hz_labels = [fmt_hz(mel_to_hz(m)) for m in mel_ticks]
        ax.set_yticks(mel_bin_indices)
        ax.set_yticklabels(hz_labels)

        # ---- metrics text panel ----
        if has_metrics:
            m = metrics[row]
            tax = fig.add_subplot(gs[row, 1])
            tax.axis("off")

            lines = [
                f"RMS:  {m['rms_db']:.1f} dBFS",
                f"SEL:  {m['sel_db']:.1f} dBFS",
            ]
            # Only show deltas for panels specified in delta_indices
            if (
                ref_metrics is not None
                and delta_indices is not None
                and row in delta_indices
            ):
                delta_rms = m["rms_db"] - ref_metrics["rms_db"]
                delta_sel = m["sel_db"] - ref_metrics["sel_db"]
                lines += [
                    "",
                    f"ΔRMS: {delta_rms:+.1f} dB",
                    f"ΔSEL: {delta_sel:+.1f} dB",
                ]

            tax.text(
                0.05,
                0.5,
                "\n".join(lines),
                transform=tax.transAxes,
                ha="left",
                va="center",
                fontsize=8,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", alpha=0.9),
            )

    # Add a single colorbar for all spectrograms to the right of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Magnitude (dB, normalized)")

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined spectrogram -> {save_path}")


def plot_combined_spectrograms_from_wavs(
    wav_paths: list[Path],
    save_path: Path,
    titles: list[str] | None = None,
    sr: int | None = None,
    ref_idx: int | None = None,
    delta_indices: list[int] | None = None,
) -> None:
    """Load several WAVs, compute their spectrograms, and make a combined plot.

    Titles default to the basename of each file if not supplied.

    If *sr* is ``None`` (the default) the sample rate is auto-detected from
    the first file and all files are loaded at that rate.

    Args:
        wav_paths: Ordered list of WAV file paths.
        save_path: Output PNG path.
        titles:    Per-panel titles; defaults to each file's basename.
        sr:        Target sample rate for loading.  ``None`` → auto-detect
                   from the first file.
        ref_idx:   Index into *wav_paths* of the panel to use as the energy
                   reference when computing ΔRMS / ΔSEL values.  Pass ``2``
                   to use the mixture as the reference.  When ``None``
                   (default) no delta values are shown.
        delta_indices: List of panel indices that should show delta values
                   relative to ref_idx. If None and ref_idx is provided, all
                   panels except ref_idx show deltas.
    """
    if sr is None:
        sr = sf.info(str(wav_paths[0])).samplerate

    specs = []
    wav_metrics = []
    used_sr: int = sr
    for p in wav_paths:
        wav, used_sr = load_wav(p, target_sr=sr)
        spec, used_sr = compute_spectrogram(wav, sr=used_sr)
        specs.append(spec)
        wav_metrics.append(compute_energy_metrics(wav, sr=used_sr))

    if titles is None:
        titles = [p.name for p in wav_paths]

    plot_combined_spectrograms(
        specs,
        titles,
        save_path,
        sr=used_sr,
        metrics=wav_metrics,
        ref_idx=ref_idx,
        delta_indices=delta_indices,
    )


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Dev shortcut — bypasses any command-line interface.  Edit wav_files,
    # save_path, titles, and ref_idx directly, then run:
    #   python demo_separation.py
    # ---------------------------------------------------------------------------
    _detected_sr = sf.info(
        "/home/bendm/Thesis/project/code/src/validation_functions/meeting_26_03/validation_examples_test_tuss/cnn/mixture_sep/mixture_coi_clean_0.wav"
    ).samplerate
    wav_files = [
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/meeting_26_03/validation_examples_test_tuss/cnn/mixture_sep/mixture_coi_clean_0.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/meeting_26_03/validation_examples_test_tuss/cnn/mixture_sep/mixture_bg_clean_0.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/meeting_26_03/validation_examples_test_tuss/cnn/mixture_sep/mixture_created_0.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/meeting_26_03/validation_examples_test_tuss/cnn/mixture_sep/mixture_separated_src0_0.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/meeting_26_03/validation_examples_test_tuss/cnn/mixture_sep/mixture_separated_src1_0.wav"
        ),
    ]
    plot_combined_spectrograms_from_wavs(
        wav_files,
        Path("meeting_26_03/combined_mixture_plane_tuss.png"),
        titles=[
            "Plane (Clean)",
            "Background (Clean)",
            "Mixture",
            "Separated Plane",
            "Separated BG",
        ],
        sr=_detected_sr,
        ref_idx=2,  # Use mixture as reference
        delta_indices=[3, 4],  # Only show deltas for the separated sources
    )
