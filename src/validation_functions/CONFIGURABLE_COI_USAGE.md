# Configurable COI Validation Guide

This guide explains how to use the configurable COI (Class of Interest) system for validating different types of classifiers (airplane, bird, etc.).

## Overview

The validation pipeline now supports multiple COI types:
- **Airplane detection** (plane, aircraft, jet, etc.)
- **Bird detection** (bird, avian, songbird, etc.)
- **Custom COI sets** (define your own)

The system automatically selects the appropriate COI synonyms based on the classifier type, ensuring correct label assignment and validation metrics.

## Quick Start

### Airplane Detection (Default)

```python
from src.validation_functions.test_pipeline import ValidationPipeline

# Create pipeline with automatic COI detection
pipeline = ValidationPipeline(base_path="/path/to/datasets")

# Load airplane classifier - COI synonyms automatically set to AIRPLANE_SYNONYMS
pipeline.load_models(
    sep_checkpoint="path/to/separator.pt",
    cls_weights="path/to/classifier.h5",
    classifier_type="plane",  # Automatic: uses AIRPLANE_SYNONYMS
)

# Run validation - uses AIRPLANE_SYNONYMS for label matching
results = pipeline.validate_all(
    split="test",
    data_csv="path/to/dataset.csv",
)
```

### Bird Detection with BirdNET

```python
from src.validation_functions.test_pipeline import ValidationPipeline

# Create pipeline
pipeline = ValidationPipeline(base_path="/path/to/datasets")

# Load BirdNET classifier - COI synonyms automatically set to BIRD_SYNONYMS
pipeline.load_models(
    sep_checkpoint="path/to/separator.pt",
    classifier_type="birdnet",  # Automatic: uses BIRD_SYNONYMS
    use_pann=False,
    use_ast=False,
)

# Run validation - uses BIRD_SYNONYMS for label matching
# Bird samples labeled as COI (1), non-bird as background (0)
results = pipeline.validate_all(
    split="test",
    data_csv="path/to/bird_dataset.csv",
)
```

### Manual COI Override

```python
from src.validation_functions.test_pipeline import ValidationPipeline
from src.label_loading.coi_labels import BIRD_SYNONYMS, AIRPLANE_SYNONYMS

# Create pipeline with custom COI synonyms
custom_synonyms = {"whale", "dolphin", "marine mammal"}

pipeline = ValidationPipeline(
    base_path="/path/to/datasets",
    coi_synonyms=custom_synonyms  # Override auto-detection
)

# Or mix existing sets
mixed_synonyms = AIRPLANE_SYNONYMS | BIRD_SYNONYMS  # Detect both planes AND birds as COI

pipeline = ValidationPipeline(
    base_path="/path/to/datasets",
    coi_synonyms=mixed_synonyms
)
```

## How COI Auto-Detection Works

The pipeline automatically maps classifier types to appropriate COI synonym sets:

| Classifier Type | COI Synonyms Used | Examples |
|----------------|-------------------|----------|
| `"plane"` | `AIRPLANE_SYNONYMS` | "plane", "airplane", "aircraft", "jet engine" |
| `"pann"` | `AIRPLANE_SYNONYMS` | (AudioSet-based airplane detection) |
| `"pann_finetuned"` | `AIRPLANE_SYNONYMS` | (Fine-tuned for airplane detection) |
| `"ast"` | `AIRPLANE_SYNONYMS` | (Audio Spectrogram Transformer) |
| `"birdnet"` | `BIRD_SYNONYMS` | "bird", "avian", "songbird", "waterfowl" |

This happens automatically in `load_models()`:

```python
# Inside load_models()
if self.coi_synonyms is None:
    self.coi_synonyms = get_coi_synonyms_for_classifier(classifier_type)
    # birdnet → BIRD_SYNONYMS
    # plane   → AIRPLANE_SYNONYMS
```

## Available COI Synonym Sets

### AIRPLANE_SYNONYMS

```python
from src.label_loading.coi_labels import AIRPLANE_SYNONYMS

# Contains 13 airplane-related terms:
{
    "plane", "planes",
    "airplane", "airplanes",
    "aeroplane", "aeroplanes",
    "aircraft",
    "fixed-wing aircraft, airplane",
    "fixed wing aircraft, airplane",
    "fixed-wing aircraft",
    "fixed wing aircraft",
    "aircraft engine",
    "jet engine",
    "propeller, airscrew",
}
```

### BIRD_SYNONYMS

```python
from src.label_loading.coi_labels import BIRD_SYNONYMS

# Contains bird-related terms:
{
    "bird", "birds",
    "avian",
    "birdsong", "bird song",
    "bird call", "bird calls",
    "bird vocalization, bird call, bird song",
    "songbird", "songbirds",
    "waterfowl",
    "raptor", "raptors",
}
```

**Note**: Extend `BIRD_SYNONYMS` in `src/label_loading/coi_labels.py` if your dataset uses different bird labels.

## What Gets Configured

When you set COI synonyms (manually or automatically), they control:

1. **Label Assignment** - Which samples are marked as COI (label=1) vs background (label=0)
2. **Contamination Filtering** - Which background samples are filtered out as contaminated
3. **Validation Metrics** - Ground truth for computing precision, recall, F1, etc.

### Example: Bird Testing

```python
# Dataset has labels: "bird", "airplane", "wind", "traffic"

pipeline = ValidationPipeline()
pipeline.load_models(classifier_type="birdnet")  # Auto-uses BIRD_SYNONYMS

# Label assignment using BIRD_SYNONYMS:
# "bird"     → label=1 (COI)
# "airplane" → label=0 (background)
# "wind"     → label=0 (background)
# "traffic"  → label=0 (background)

# Contamination check:
# Background samples with "bird" in orig_label → FILTERED OUT

results = pipeline.validate_all(...)
# Metrics computed against bird detection ground truth ✅
```

## Dataset Preparation

Your CSV should have labels compatible with the COI synonyms you're using:

### For Airplane Testing

```csv
filename,label,orig_label,split
audio1.wav,1,"airplane",train
audio2.wav,0,"wind",train
audio3.wav,1,"jet engine",test
audio4.wav,0,"traffic",test
```

### For Bird Testing

```csv
filename,label,orig_label,split
audio1.wav,1,"bird",train
audio2.wav,0,"wind",train
audio3.wav,1,"songbird",test
audio4.wav,0,"traffic",test
```

**Important**: The `orig_label` column should contain the raw label string. If your dataset uses different terminology, either:
1. Add your terms to the appropriate synonym set in `coi_labels.py`, OR
2. Create a custom synonym set and pass it to `ValidationPipeline(coi_synonyms=...)`

## Advanced Usage

### Custom COI for Specific Species

```python
from src.validation_functions.test_pipeline import ValidationPipeline

# Detect only crows
crow_synonyms = {
    "crow",
    "american crow",
    "corvus brachyrhynchos",
    "corvid",
}

pipeline = ValidationPipeline(coi_synonyms=crow_synonyms)
pipeline.load_models(classifier_type="birdnet")  # Will use crow_synonyms, not BIRD_SYNONYMS

results = pipeline.validate_all(...)
```

### Multi-Class to Binary Mapping

```python
# Detect "transportation sounds" as COI
transport_synonyms = {
    "airplane", "aircraft", "plane",
    "train", "railway",
    "car", "vehicle", "traffic",
    "ship", "boat",
}

pipeline = ValidationPipeline(coi_synonyms=transport_synonyms)
```

## Validation Output

The validation results clearly indicate which COI synonyms were used:

```
Loading primary birdnet classifier...
  Classifier sample rate: 48000 Hz
  Auto-detected COI synonyms: BIRD_SYNONYMS (11 terms)

Dataset Statistics:
  COI samples:        245 (birds detected using BIRD_SYNONYMS)
  Background samples: 1523 clean
  
Confusion Matrix:  TN=1480  FP=43
                   FN=12    TP=233

Bird Detection Performance:
  Accuracy:  0.9684    Precision: 0.9446
  Recall:    0.9510    F1-Score:  0.9478
```

## Testing Airplane and Bird Separately

To test both airplane and bird detection in the same codebase:

```python
# Test 1: Airplane detection
pipeline_plane = ValidationPipeline()
pipeline_plane.load_models(classifier_type="plane")
results_plane = pipeline_plane.validate_all(data_csv="airplane_dataset.csv")

# Test 2: Bird detection
pipeline_bird = ValidationPipeline()
pipeline_bird.load_models(classifier_type="birdnet")
results_bird = pipeline_bird.validate_all(data_csv="bird_dataset.csv")
```

## Troubleshooting

### Labels Not Detected as COI

**Problem**: Bird samples labeled as background even though using BirdNET.

**Solution**: Check if your label names match `BIRD_SYNONYMS`:

```python
from src.label_loading.coi_labels import BIRD_SYNONYMS, is_coi_label

# Test your labels
test_label = "robin"  # Your dataset label
print(is_coi_label(test_label, BIRD_SYNONYMS))  # False - not in BIRD_SYNONYMS

# Solution 1: Add to BIRD_SYNONYMS in coi_labels.py
# Solution 2: Use custom synonyms
custom_bird_syns = BIRD_SYNONYMS | {"robin", "sparrow", "warbler"}
pipeline = ValidationPipeline(coi_synonyms=custom_bird_syns)
```

### Contaminated Background Samples

**Problem**: Warning about contaminated backgrounds.

```
⚠️  CONTAMINATION DETECTED: 23 background samples
   contain COI synonyms in orig_label and will be EXCLUDED
```

**Explanation**: Background samples (label=0) have COI terms in `orig_label`. These are automatically filtered to prevent contamination. This is expected behavior when datasets have mislabeled samples.

### Wrong Metrics for Bird Testing

**Problem**: Metrics don't make sense when testing BirdNET.

**Solution**: Ensure you're using the right classifier type:

```python
# ❌ Wrong - uses AIRPLANE_SYNONYMS by default
pipeline.load_models(classifier_type="plane")  # Testing on bird dataset!

# ✅ Correct - uses BIRD_SYNONYMS
pipeline.load_models(classifier_type="birdnet")  # Testing on bird dataset
```

## Summary

The configurable COI system enables:
- ✅ Airplane detection with airplane-specific labels
- ✅ Bird detection with bird-specific labels  
- ✅ Custom COI sets for any detection task
- ✅ Automatic COI selection based on classifier type
- ✅ Manual override when needed
- ✅ Proper validation metrics for each task

No code changes needed for standard airplane/bird testing - just specify the right `classifier_type` in `load_models()`!
