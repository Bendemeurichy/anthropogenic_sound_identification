# anthropogenic_sound_seperation and filtering

Repo for master thesis in `machine learning for biodiversity`

## Features

- **YAMNet Finetuning**: Finetune the YAMNet model for airplane sound detection using transfer learning

## Installation

```bash
pip install tensorflow numpy pandas librosa resampy soundfile tqdm matplotlib
```

## YAMNet Airplane Detection Finetuning

This repository includes a module for finetuning YAMNet for airplane sound detection. The finetuning approach uses transfer learning: features are extracted from audio using the pre-trained YAMNet model, and a classifier is trained on top of these features.

### Quick Start

```python
from src.finetuning import FinetuneConfig, train

config = FinetuneConfig(
    path_yamnet="yamnet_planes",
    path_data_train="/path/to/training_data",
    path_data_csv="/path/to/audio_paths.csv",
    path_save="/path/to/save_models",
    epochs=1000,
    learning_rate=0.001,
)

result = train(config)
print(f"Model saved to: {result['path_model']}")
```

### Command Line Interface

```bash
python -m src.finetuning.cli \
    --path-yamnet yamnet_planes \
    --path-data-train /path/to/training_data \
    --path-data-csv /path/to/audio_paths.csv \
    --path-save /path/to/save_models \
    --epochs 1000 \
    --learning-rate 0.001
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path_yamnet` | (required) | Path to yamnet_planes directory |
| `path_data_train` | (required) | Path to training audio data |
| `path_save` | (required) | Path to save models |
| `path_data_csv` | "" | CSV file with audio paths and labels |
| `classes` | ["not_plane", "plane"] | Class names |
| `patch_hop_seconds` | 0.096 | Time hop between patches |
| `min_sample_seconds` | 1.0 | Minimum audio duration |
| `max_sample_seconds` | 1000.0 | Maximum audio duration |
| `num_augmentations` | [0, 0] | Augmentations per class |
| `num_hidden` | [1024] | Hidden layer sizes |
| `num_classes` | 2 | Number of output classes |
| `optimizer_type` | "SGD" | Optimizer ("SGD" or "Adam") |
| `learning_rate` | 0.001 | Learning rate |
| `epochs` | 10000 | Maximum training epochs |
| `validation_split` | 0.1 | Validation data fraction |

### Data Format

The training data should be organized with audio files in class-specific folders:

```
training_data/
├── not_plane/
│   ├── audio1.wav
│   └── audio2.wav
└── plane/
    ├── audio3.wav
    └── audio4.wav
```

Alternatively, provide a CSV file with two columns: audio path and class label.

### Feature Extraction Methods

- **Method 0**: Extract features in memory and train directly
- **Method 1**: Extract features to disk first, then load for training (recommended for large datasets)

### Data Augmentation

Supported augmentations (applied randomly during feature extraction):
- Time stretching
- Resampling
- Volume adjustment
- Random noise injection
