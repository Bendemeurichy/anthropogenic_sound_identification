#!/usr/bin/env python3
"""
Inference script for the PlaneClassifier model.
Load a trained model and make predictions on audio files.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model_loader import load_trained_model
from config import TrainingConfig
from helpers import load_audio


class PlaneClassifierInference:
    """Wrapper for easy inference with PlaneClassifier model"""

    def __init__(self, weights_path: str, config: TrainingConfig = None):
        """Initialize inference with trained model.

        Args:
            weights_path: Path to .weights.h5 file
            config: TrainingConfig (must match training configuration)
        """
        self.config = config if config is not None else TrainingConfig()
        self.model = load_trained_model(weights_path, self.config)

    def predict_file(self, audio_path: str, threshold: float = 0.5):
        """Predict whether audio file contains a plane.

        Args:
            audio_path: Path to audio file (WAV format recommended)
            threshold: Classification threshold (default 0.5)

        Returns:
            dict with keys:
                - 'prediction': 'plane' or 'no_plane'
                - 'confidence': probability score
                - 'logit': raw model output
        """
        # Load audio
        waveform = load_audio(
            audio_path,
            sample_rate=self.config.sample_rate,
            duration=self.config.audio_duration,
        )

        # Add batch dimension
        waveform_batch = tf.expand_dims(waveform, 0)

        # Get prediction
        logit = self.model(waveform_batch, training=False)
        probability = tf.sigmoid(logit).numpy()[0, 0]

        prediction = "plane" if probability >= threshold else "no_plane"

        return {
            "prediction": prediction,
            "confidence": float(probability),
            "logit": float(logit.numpy()[0, 0]),
        }

    def predict_waveform(self, waveform: np.ndarray, threshold: float = 0.5):
        """Predict from raw waveform array.

        Args:
            waveform: Audio waveform as numpy array or tensor
            threshold: Classification threshold

        Returns:
            dict with prediction results
        """
        # Ensure correct shape and type
        if len(waveform.shape) == 1:
            waveform = tf.expand_dims(waveform, 0)

        # Get prediction
        logit = self.model(waveform, training=False)
        probability = tf.sigmoid(logit).numpy()[0, 0]

        prediction = "plane" if probability >= threshold else "no_plane"

        return {
            "prediction": prediction,
            "confidence": float(probability),
            "logit": float(logit.numpy()[0, 0]),
        }

    def predict_batch(self, audio_paths: list, threshold: float = 0.5):
        """Predict on multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            threshold: Classification threshold

        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict_file(path, threshold)
                result["file"] = path
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append(
                    {
                        "file": path,
                        "prediction": "error",
                        "confidence": None,
                        "logit": None,
                        "error": str(e),
                    }
                )
        return results


def main():
    """Command-line interface for inference"""
    parser = argparse.ArgumentParser(
        description="Run inference with trained PlaneClassifier model"
    )
    parser.add_argument(
        "weights_path", type=str, help="Path to model weights file (.weights.h5)"
    )
    parser.add_argument(
        "audio_files", type=str, nargs="+", help="Audio file(s) to classify"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )

    args = parser.parse_args()

    # Check weights file exists
    if not Path(args.weights_path).exists():
        print(f"Error: Weights file not found: {args.weights_path}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    classifier = PlaneClassifierInference(args.weights_path)
    print()

    # Run predictions
    for audio_file in args.audio_files:
        if not Path(audio_file).exists():
            print(f"Warning: File not found: {audio_file}")
            continue

        print(f"Processing: {audio_file}")
        result = classifier.predict_file(audio_file, args.threshold)

        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Logit: {result['logit']:.4f}")
        print()


if __name__ == "__main__":
    main()
