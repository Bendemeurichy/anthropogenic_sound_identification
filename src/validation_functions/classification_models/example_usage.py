#!/usr/bin/env python3
"""
Example usage of the unified classifier interface.

This script demonstrates how to use different classifiers (plane, PANN, AST, BirdNET)
with the same consistent interface.
"""

import sys
from pathlib import Path
import torch
import torchaudio

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation_functions.classification_models.Classifier import create_classifier


def load_audio_for_classifier(audio_path: str, classifier) -> torch.Tensor:
    """Load and prepare audio for a classifier.
    
    Args:
        audio_path: Path to audio file
        classifier: Classifier instance with sample_rate property
        
    Returns:
        Audio waveform tensor at the classifier's expected sample rate
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    
    # Resample if needed
    if sr != classifier.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, classifier.sample_rate)
        waveform = resampler(waveform)
    
    return waveform


def main():
    """Demonstrate using different classifiers with the same interface."""
    
    # Example audio file path (replace with your own)
    audio_path = "path/to/your/audio.wav"
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        print("\nPlease update the audio_path variable with a valid audio file.")
        return
    
    print("=" * 70)
    print("Unified Classifier Interface Demo")
    print("=" * 70)
    
    # Example 1: Plane Classifier
    print("\n1. PlaneClassifier (Custom CNN for airplane detection)")
    print("-" * 70)
    
    weights_path = "path/to/plane_model.weights.h5"  # Update this path
    
    if Path(weights_path).exists():
        plane_clf = create_classifier(
            "plane",
            weights_path=weights_path,
            threshold=0.5,
            device="cpu"  # or "cuda" for GPU
        )
        
        waveform = load_audio_for_classifier(audio_path, plane_clf)
        prediction, confidence = plane_clf(waveform)
        
        print(f"Sample rate: {plane_clf.sample_rate} Hz")
        print(f"Prediction: {'AIRPLANE DETECTED' if prediction == 1 else 'No airplane'}")
        print(f"Confidence: {confidence:.4f}")
    else:
        print(f"Skipping (weights not found: {weights_path})")
    
    # Example 2: PANN Classifier
    print("\n2. PANN (AudioSet-based classifier)")
    print("-" * 70)
    
    try:
        pann_clf = create_classifier(
            "pann",
            device="cpu",
            threshold=0.5,
            positive_labels=[
                "Fixed-wing aircraft, airplane",
                "Aircraft",
                "Jet aircraft"
            ]
        )
        
        waveform = load_audio_for_classifier(audio_path, pann_clf)
        prediction, confidence = pann_clf(waveform)
        
        print(f"Sample rate: {pann_clf.sample_rate} Hz")
        print(f"Prediction: {'AIRCRAFT DETECTED' if prediction == 1 else 'No aircraft'}")
        print(f"Confidence: {confidence:.4f}")
    except ImportError as e:
        print(f"Skipping (dependency not installed: {e})")
    
    # Example 3: AST Classifier
    print("\n3. AST (Audio Spectrogram Transformer)")
    print("-" * 70)
    
    try:
        ast_clf = create_classifier(
            "ast",
            device="cpu",
            threshold=0.5,
            positive_labels=[
                "Fixed-wing aircraft, airplane",
                "Aircraft"
            ]
        )
        
        waveform = load_audio_for_classifier(audio_path, ast_clf)
        prediction, confidence = ast_clf(waveform)
        
        print(f"Sample rate: {ast_clf.sample_rate} Hz")
        print(f"Prediction: {'AIRCRAFT DETECTED' if prediction == 1 else 'No aircraft'}")
        print(f"Confidence: {confidence:.4f}")
    except ImportError as e:
        print(f"Skipping (dependency not installed: {e})")
    
    # Example 4: BirdNET Classifier
    print("\n4. BirdNET (Bird species detection)")
    print("-" * 70)
    
    try:
        bird_clf = create_classifier(
            "birdnet",
            device="cpu",
            threshold=0.5,
            detect_any_bird=True  # Detect any bird species
        )
        
        waveform = load_audio_for_classifier(audio_path, bird_clf)
        prediction, confidence = bird_clf(waveform)
        
        print(f"Sample rate: {bird_clf.sample_rate} Hz")
        print(f"Prediction: {'BIRD DETECTED' if prediction == 1 else 'No bird'}")
        print(f"Confidence: {confidence:.4f}")
    except ImportError as e:
        print(f"Skipping (BirdNET not installed: {e})")
    
    # Example 5: Using classifiers interchangeably
    print("\n5. Demonstrating Interchangeable Use")
    print("-" * 70)
    print("All classifiers follow the same protocol:")
    print("  - classifier(waveform) -> (prediction, confidence)")
    print("  - classifier.sample_rate -> int")
    print("\nThis allows you to swap classifiers in your pipeline without")
    print("changing the rest of your code!")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
