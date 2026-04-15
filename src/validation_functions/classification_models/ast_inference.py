"""Wrapper for AST (Audio Spectrogram Transformer) inference.

Uses the Hugging Face Transformers library to load the pre-trained AST model
fine-tuned on AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593).

Example usage:
    from ast_inference import load_ast_model, run_ast_inference, AST_SAMPLE_RATE

    extractor, model = load_ast_model(device="cuda")
    pred, conf = run_ast_inference(
        extractor, model, waveform, AST_SAMPLE_RATE,
        ["Fixed-wing aircraft, airplane", "Jet aircraft"],
        device="cuda",
    )
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.common.audio_utils import create_high_quality_resampler

# The MIT AST model was fine-tuned on AudioSet with 16 kHz audio.
AST_SAMPLE_RATE: int = 16_000

# HuggingFace model identifier.
_MODEL_ID: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Default positive-class AudioSet labels for aircraft / airplane detection.
# Override or replace this list as needed.
DEFAULT_AST_POSITIVE_LABELS: List[str] = [
    "Fixed-wing aircraft, airplane",
    "Aircraft",
    "Jet aircraft",
    "Propeller, airscrew",
    "Turboprop, small aircraft",
]


def load_ast_model(
    device: str = "cpu",
) -> Tuple[AutoFeatureExtractor, AutoModelForAudioClassification]:
    """Load the AST feature extractor and classification model.

    Args:
        device: PyTorch device string (e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``).

    Returns:
        ``(extractor, model)`` — the feature extractor and the model moved to
        *device* and set to eval mode.
    """
    print(f"Loading AST model {_MODEL_ID!r} on device={device!r} …", file=sys.stderr)
    extractor = AutoFeatureExtractor.from_pretrained(_MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(_MODEL_ID)
    model = model.to(device)
    model.eval()
    print("AST model loaded.", file=sys.stderr)
    return extractor, model


def run_ast_inference(
    extractor: AutoFeatureExtractor,
    model: AutoModelForAudioClassification,
    waveform: torch.Tensor,
    sample_rate: int,
    positive_labels: List[str],
    device: str = "cpu",
    threshold: float = 0.5,
) -> Tuple[int, float]:
    """Run AST inference and return a binary prediction.

    The function resamples the input to :data:`AST_SAMPLE_RATE` when needed,
    runs the Audio Spectrogram Transformer, and aggregates per-class sigmoid
    scores for all labels listed in *positive_labels*.  The maximum score over
    those labels is used as the confidence value; the prediction is positive (1)
    when that score meets or exceeds *threshold*.

    AudioSet is a multi-label problem so sigmoid (not softmax) activations are
    used for the output probabilities.

    Args:
        extractor: The HuggingFace feature extractor for the AST model.
        model: A loaded :class:`~transformers.AutoModelForAudioClassification`
            instance.
        waveform: 1-D float :class:`torch.Tensor` at ``sample_rate`` Hz.
        sample_rate: Sample rate of *waveform* in Hz.
        positive_labels: AudioSet label names that should be treated as the
            positive class.  Unrecognised names are silently ignored with a
            warning printed to stderr.
        device: Device on which the model lives (must match where *model* was
            placed by :func:`load_ast_model`).
        threshold: Confidence threshold for a positive decision (default 0.5).

    Returns:
        ``(prediction, confidence)`` where *prediction* is 1 (positive) or 0
        (negative) and *confidence* is the maximum sigmoid score across all
        matched positive labels (0.0 when no labels are matched).
    """
    wav = waveform.detach().cpu()

    # Resample to AST's expected 16 kHz when necessary.
    if sample_rate != AST_SAMPLE_RATE:
        resampler = create_high_quality_resampler(
            orig_sr=sample_rate, target_sr=AST_SAMPLE_RATE
        )
        wav = resampler(wav)

    waveform_np = wav.numpy()

    # The HuggingFace extractor handles mel-filterbank / normalisation.
    inputs = extractor(
        waveform_np,
        sampling_rate=AST_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, num_labels)

    # Multi-label AudioSet classification → sigmoid probabilities.
    probs = torch.sigmoid(logits)[0].cpu()  # (num_labels,)

    # Build a case-insensitive label → index mapping from the model config.
    id2label = model.config.id2label  # dict[int, str]
    label2id: dict = {v.lower(): k for k, v in id2label.items()}

    pos_indices: List[int] = []
    for label in positive_labels:
        key = label.lower()
        if key in label2id:
            pos_indices.append(label2id[key])
        else:
            print(
                f"[AST] Warning: label {label!r} not found in model label list – skipping.",
                file=sys.stderr,
            )

    if pos_indices:
        pos_score = float(max(probs[i].item() for i in pos_indices))
    else:
        pos_score = 0.0

    prediction = 1 if pos_score >= threshold else 0
    return prediction, pos_score
