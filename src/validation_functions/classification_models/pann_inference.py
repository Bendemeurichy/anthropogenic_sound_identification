"""Wrapper for PANN (Pretrained Audio Neural Networks) inference.

Adapted from the panns_inference library.

Example usage:
    import librosa
    from pann_inference import load_pann_model, run_pann_inference, PANN_SAMPLE_RATE

    audio, _ = librosa.load("audio.wav", sr=PANN_SAMPLE_RATE, mono=True)
    waveform = torch.tensor(audio)

    model = load_pann_model(device="cuda")
    pred, conf = run_pann_inference(model, waveform, PANN_SAMPLE_RATE, ["Aircraft", "Jet aircraft"])
"""

import sys
from typing import List, Tuple

import torch
import torchaudio
from panns_inference import AudioTagging
from panns_inference import (
    labels as AUDIOSET_LABELS,  # list of 527 AudioSet class names
)

# PANN models were trained on audio resampled to 32 kHz.
PANN_SAMPLE_RATE: int = 32_000

# Default positive-class AudioSet labels for aircraft / airplane detection.
# Override or replace this list as needed.
DEFAULT_PANN_POSITIVE_LABELS: List[str] = [
    "Fixed-wing aircraft, airplane",
    "Aircraft",
    "Jet aircraft",
    "Propeller, airscrew",
    "Turboprop, small aircraft",
]


def load_pann_model(device: str = "cuda") -> AudioTagging:
    """Load the PANN CNN14 AudioTagging model.

    Args:
        device: PyTorch device string (e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``).

    Returns:
        A ready-to-use :class:`panns_inference.AudioTagging` instance.
    """
    print(f"Loading PANN AudioTagging model on device={device!r} …", file=sys.stderr)
    at = AudioTagging(checkpoint_path=None, device=device)
    print("PANN model loaded.", file=sys.stderr)
    return at


def run_pann_inference(
    model: AudioTagging,
    waveform: torch.Tensor,
    sample_rate: int,
    positive_labels: List[str],
    threshold: float = 0.5,
) -> Tuple[int, float]:
    """Run PANN inference and return a binary prediction.

    The function resamples the input to :data:`PANN_SAMPLE_RATE` when needed,
    runs the AudioTagging model, and then aggregates the per-class sigmoid scores
    for all labels listed in *positive_labels*.  The maximum score over those
    labels is used as the confidence value; the prediction is positive (1) when
    that score meets or exceeds *threshold*.

    Args:
        model: A loaded :class:`panns_inference.AudioTagging` instance.
        waveform: 1-D float :class:`torch.Tensor` at ``sample_rate`` Hz.
        sample_rate: Sample rate of *waveform* in Hz.
        positive_labels: AudioSet label names that should be treated as the
            positive class.  Unrecognised names are silently ignored.
        threshold: Confidence threshold for a positive decision (default 0.5).

    Returns:
        ``(prediction, confidence)`` where *prediction* is 1 (positive) or 0
        (negative) and *confidence* is the maximum sigmoid score across all
        matched positive labels (0.0 when no labels are matched).
    """
    wav = waveform.detach().cpu()

    # Resample to PANN's expected 32 kHz when necessary.
    if sample_rate != PANN_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=PANN_SAMPLE_RATE
        )
        wav = resampler(wav)

    # panns_inference expects a NumPy array of shape (batch, samples).
    audio_np = wav.numpy()[None, :]  # (1, T)

    clipwise_output, _ = model.inference(audio_np)
    scores = clipwise_output[0]  # (527,) – one sigmoid score per AudioSet class

    # Resolve label names to their integer indices in the AudioSet ontology.
    pos_indices: List[int] = []
    for label in positive_labels:
        if label in AUDIOSET_LABELS:
            pos_indices.append(AUDIOSET_LABELS.index(label))
        else:
            print(
                f"[PANN] Warning: label {label!r} not found in AudioSet label list – skipping.",
                file=sys.stderr,
            )

    if pos_indices:
        pos_score = float(max(scores[i] for i in pos_indices))
    else:
        pos_score = 0.0

    prediction = 1 if pos_score >= threshold else 0
    return prediction, pos_score
