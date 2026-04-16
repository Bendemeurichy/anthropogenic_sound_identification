"""
Inference script for the PANN PlaneClassifier model.
Load a trained model and make predictions on audio files.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from model_loader import load_trained_model
from config import TrainingConfig
from common.audio_utils import create_high_quality_resampler


class PlaneClassifierInference:
    """Wrapper for easy inference with PANN PlaneClassifier model"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: TrainingConfig = None,
        device: str = None
    ):
        """
        Initialize inference with trained model.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            config: TrainingConfig (must match training configuration)
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.config = config if config is not None else TrainingConfig()
        self.model = load_trained_model(checkpoint_path, training_config=self.config, device=device)
        self.model.eval()
        
        print(f"Model loaded on {device}")
    
    def predict_file(self, audio_path: str, threshold: float = 0.5):
        """
        Predict whether audio file contains a plane.
        
        Args:
            audio_path: Path to audio file
            threshold: Classification threshold (default 0.5)
            
        Returns:
            dict with keys:
                - 'prediction': 'plane' or 'no_plane'
                - 'confidence': probability score
                - 'logit': raw model output
        """
        # Load audio
        waveform = self._load_audio(audio_path)
        
        # Add batch dimension
        waveform_batch = waveform.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logit = self.model(waveform_batch)
            probability = torch.sigmoid(logit).cpu().numpy()[0, 0]
        
        prediction = "plane" if probability >= threshold else "no_plane"
        
        return {
            "prediction": prediction,
            "confidence": float(probability),
            "logit": float(logit.cpu().numpy()[0, 0]),
        }
    
    def predict_waveform(self, waveform: torch.Tensor, threshold: float = 0.5):
        """
        Predict from raw waveform tensor.
        
        Args:
            waveform: Audio waveform tensor (should be at 32kHz)
            threshold: Classification threshold
            
        Returns:
            dict with prediction results
        """
        # Ensure correct shape
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logit = self.model(waveform)
            probability = torch.sigmoid(logit).cpu().numpy()[0, 0]
        
        prediction = "plane" if probability >= threshold else "no_plane"
        
        return {
            "prediction": prediction,
            "confidence": float(probability),
            "logit": float(logit.cpu().numpy()[0, 0]),
        }
    
    def predict_batch(self, audio_paths: list, threshold: float = 0.5):
        """
        Predict on multiple audio files.
        
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
                results.append({
                    "file": path,
                    "prediction": "error",
                    "confidence": None,
                    "logit": None,
                    "error": str(e),
                })
        return results
    
    def _load_audio(self, file_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Preprocessed waveform tensor
        """
        target_samples = int(self.config.sample_rate * self.config.audio_duration)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            resampler = create_high_quality_resampler(
                orig_sr=sample_rate,
                target_sr=self.config.sample_rate
            )
            waveform = resampler(waveform)
        
        # Normalize length (pad or center crop)
        current_len = len(waveform)
        if current_len < target_samples:
            # Pad
            padding = target_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > target_samples:
            # Center crop
            start = (current_len - target_samples) // 2
            waveform = waveform[start : start + target_samples]
        
        return waveform


def main():
    """Command-line interface for inference"""
    parser = argparse.ArgumentParser(
        description="Run inference with trained PANN PlaneClassifier model"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to model checkpoint file (.pth)"
    )
    parser.add_argument(
        "audio_files",
        type=str,
        nargs="+",
        help="Audio file(s) to classify"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda or cpu). Auto-detect if not specified."
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Load model
    print("Loading model...")
    classifier = PlaneClassifierInference(args.checkpoint_path, device=args.device)
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
