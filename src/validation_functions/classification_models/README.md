# Unified Classifier Interface

This directory contains a unified interface for audio classification models used in the validation pipeline.

## Overview

All classifiers follow a common protocol:
- **Input**: Audio waveform as a 1D `torch.Tensor`
- **Output**: Tuple of `(prediction: int, confidence: float)`
  - `prediction`: Binary class (0 or 1)
  - `confidence`: Score in range [0.0, 1.0]

## Supported Classifiers

### 1. PlaneClassifier (`plane`)
- **Description**: Custom CNN with YAMNet backbone for airplane detection
- **Framework**: TensorFlow
- **Sample Rate**: 16,000 Hz (configurable)
- **Required**: `weights_path` parameter pointing to `.weights.h5` file

### 2. PANN (`pann`)
- **Description**: Pretrained Audio Neural Network for AudioSet classification
- **Framework**: PyTorch
- **Sample Rate**: 32,000 Hz
- **Dependencies**: `panns_inference`

### 3. AST (`ast`)
- **Description**: Audio Spectrogram Transformer for AudioSet
- **Framework**: HuggingFace Transformers
- **Sample Rate**: 16,000 Hz
- **Dependencies**: `transformers`

### 4. BirdNET (`birdnet`)
- **Description**: Bird species acoustic classifier (6,522 species)
- **Framework**: TensorFlow/TFLite
- **Sample Rate**: 48,000 Hz
- **Dependencies**: `birdnet`
- **Installation**: `pip install birdnet`

## Quick Start

```python
from Classifier import create_classifier
import torch

# Create any classifier using the factory function
classifier = create_classifier(
    "plane",  # or "pann", "ast", "birdnet"
    weights_path="path/to/model.weights.h5",  # for plane classifier
    threshold=0.5,
    device="cuda"  # or "cpu"
)

# Load and prepare audio at the correct sample rate
# (See example_usage.py for full audio loading code)
waveform = torch.randn(classifier.sample_rate * 3)  # 3 seconds

# Run classification
prediction, confidence = classifier(waveform)

print(f"Prediction: {prediction}")  # 0 or 1
print(f"Confidence: {confidence:.4f}")  # 0.0 to 1.0
```

## Factory Function: `create_classifier()`

```python
create_classifier(
    classifier_type: str,
    device: str = "cpu",
    threshold: float = 0.5,
    **kwargs
) -> AudioClassifier
```

### Common Parameters
- `classifier_type`: One of `"plane"`, `"pann"`, `"ast"`, `"birdnet"`
- `device`: PyTorch device string (`"cpu"`, `"cuda"`, `"cuda:0"`, etc.)
- `threshold`: Classification threshold (default: 0.5)

### Classifier-Specific Parameters

#### PlaneClassifier
```python
classifier = create_classifier(
    "plane",
    weights_path="model.weights.h5",  # Required
    config=TrainingConfig(...),  # Optional
    device="cuda"
)
```

#### PANN
```python
classifier = create_classifier(
    "pann",
    device="cuda",
    positive_labels=[  # Optional, defaults to aircraft labels
        "Fixed-wing aircraft, airplane",
        "Aircraft",
        "Jet aircraft"
    ]
)
```

#### AST
```python
classifier = create_classifier(
    "ast",
    device="cuda",
    positive_labels=[  # Optional, defaults to aircraft labels
        "Fixed-wing aircraft, airplane",
        "Aircraft"
    ]
)
```

#### BirdNET
```python
classifier = create_classifier(
    "birdnet",
    device="cpu",  # BirdNET works best on CPU
    detect_any_bird=True,  # Detect any bird species
    target_species=None,  # Or specify list of species
    model_version="2.4",  # BirdNET version
    backend="tf"  # "tf" or "tflite"
)
```

## File Structure

```
classification_models/
├── Classifier.py              # Factory function and protocol definition
├── plane_wrapper.py           # PlaneClassifier wrapper
├── pann_wrapper.py            # PANN wrapper
├── ast_wrapper.py             # AST wrapper
├── birdnet_wrapper.py         # BirdNET wrapper
├── pann_inference.py          # Original PANN implementation
├── ast_inference.py           # Original AST implementation
├── plane_clasifier/           # Original PlaneClassifier
│   └── inference.py
├── example_usage.py           # Example script
└── README.md                  # This file
```

## Using in Validation Pipeline

The `test_pipeline.py` has been updated to support all classifiers:

```python
from test_pipeline import ValidationPipeline

pipeline = ValidationPipeline()

# Use plane classifier (default)
pipeline.load_models(
    sep_checkpoint="path/to/separator.pt",
    cls_weights="path/to/plane_model.weights.h5",
    classifier_type="plane"  # or "pann", "ast", "birdnet"
)

# Or use BirdNET for bird detection
pipeline.load_models(
    sep_checkpoint="path/to/separator.pt",
    classifier_type="birdnet"
)

# Run validation
metrics = pipeline.validate_clean(df_coi, df_bg, use_separation=True)
```

## Protocol Definition

All classifiers implement the `AudioClassifier` protocol:

```python
from typing import Tuple, Protocol
import torch

class AudioClassifier(Protocol):
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Classify audio waveform."""
        ...
    
    @property
    def sample_rate(self) -> int:
        """Expected sample rate for input."""
        ...
```

This protocol allows type checking and ensures all classifiers are interchangeable.

## Installation

### PlaneClassifier
Already included in the project. Requires TensorFlow.

### PANN
```bash
pip install panns_inference
```

### AST
```bash
pip install transformers
```

### BirdNET
```bash
pip install birdnet

# For GPU support (Linux only):
pip install birdnet[and-cuda]
```

## Examples

See `example_usage.py` for complete examples of using each classifier type.

## Advanced Usage

### Custom Configuration

For PlaneClassifier, you can provide a custom config:

```python
from plane_clasifier.config import TrainingConfig

config = TrainingConfig(
    sample_rate=16000,
    audio_duration=3.0,
    # ... other parameters
)

classifier = create_classifier(
    "plane",
    weights_path="model.weights.h5",
    config=config
)
```

### Using Multiple Classifiers

You can load and use multiple classifiers simultaneously:

```python
plane_clf = create_classifier("plane", weights_path="plane.h5")
bird_clf = create_classifier("birdnet")

# Classify with both
plane_pred, plane_conf = plane_clf(waveform_16k)
bird_pred, bird_conf = bird_clf(waveform_48k)  # Different sample rates!
```

### Auxiliary Classifiers in Pipeline

The validation pipeline supports auxiliary classifiers for comparison:

```python
pipeline.load_models(
    classifier_type="plane",  # Primary classifier
    cls_weights="plane.weights.h5",
    use_pann=True,  # Load PANN as auxiliary
    use_ast=True,   # Load AST as auxiliary
    use_birdnet=True  # Load BirdNET as auxiliary
)

# Use different classifiers
metrics_plane = pipeline.validate_clean(df_coi, df_bg, classify_fn=pipeline._classify)
metrics_pann = pipeline.validate_clean(df_coi, df_bg, classify_fn=pipeline._classify_pann)
metrics_bird = pipeline.validate_clean(df_coi, df_bg, classify_fn=pipeline._classify_birdnet)
```

## Notes

- **Sample Rates**: Each classifier expects a specific sample rate. Ensure your audio is resampled correctly.
- **GPU Support**: PANN and AST support GPU. BirdNET TensorFlow backend supports GPU (Linux only). PlaneClassifier uses TensorFlow's automatic device placement.
- **Threshold**: The threshold parameter controls the decision boundary. Higher = more conservative (fewer positives).
- **BirdNET**: Returns positive (1) when ANY bird species is detected above threshold (configurable).

## Troubleshooting

### Import Errors
If you get import errors, ensure all dependencies are installed in your virtual environment.

### Sample Rate Mismatch
Each classifier expects a specific sample rate. Check `classifier.sample_rate` and resample accordingly.

### Memory Issues
For large audio files, process in chunks. BirdNET and other models handle this automatically.

### GPU Not Working
- Ensure CUDA is installed correctly
- Check device placement: `device="cuda"` or `device="cuda:0"`
- Some models (like BirdNET TFLite) only support CPU

## Contributing

To add a new classifier:

1. Create a wrapper class implementing the `AudioClassifier` protocol
2. Add it to `create_classifier()` factory function in `Classifier.py`
3. Update this README
4. Add usage example to `example_usage.py`
