# YAMNet Fine-tuning Guide

## How Fine-tuning Works in This Implementation

### Phase 1: Train Classifier Head Only
During Phase 1, only the classification head (dense layers on top of YAMNet) is trained. YAMNet embeddings are computed but the YAMNet weights are not updated.

### Phase 2: Fine-tune Entire Model
During Phase 2, **both** YAMNet and the classifier head are trained with a lower learning rate. This allows the model to adapt the audio embeddings to better discriminate between plane and non-plane sounds.

## Technical Details

### Why Save Weights Instead of Full Model?
TensorFlow Hub models (like YAMNet) cannot be serialized using Keras's standard model saving format. Therefore, we save only the **weights** (.weights.h5) instead of the full model (.keras).

### Does Fine-tuning Actually Work?
**Yes!** Even though TensorFlow Hub models don't support the `.trainable` attribute in the standard Keras way, all variables in the hub model are included in `model.trainable_variables` by default. When you compile and train the model with `model.fit()`, the optimizer will update ALL trainable variables, including those in YAMNet.

### Verification
You can verify that YAMNet is being fine-tuned by:

1. **Check trainable variables count:**
```python
print(f"Total trainable variables: {len(model.trainable_variables)}")
```
In Phase 1, this should be ~3-6 variables (classifier head only).
In Phase 2, this should be ~100+ variables (YAMNet + classifier head).

2. **Monitor YAMNet gradients:**
During Phase 2 training, you can verify gradients are flowing through YAMNet.

3. **Compare performance:**
Phase 2 should show improvement over Phase 1 if fine-tuning is working.

## Loading a Trained Model

```python
from model_loader import load_trained_model
from config import TrainingConfig

# Load configuration (must match training config)
config = TrainingConfig()

# Load model from weights
model = load_trained_model("checkpoints/final_model.weights.h5", config)

# Use for inference
predictions = model.predict(test_dataset)
```

## Checkpoint Files

After training, you'll have:
- `best_model_phase1.weights.h5` - Best model from Phase 1
- `best_model_phase2.weights.h5` - Best model from Phase 2 (fine-tuned)
- `final_model.weights.h5` - Final model after all training
- `training_phase1.csv` - Phase 1 training metrics
- `training_phase2.csv` - Phase 2 training metrics

## Important Notes

1. **Architecture must match**: When loading weights, the model architecture (hidden units, dropout rates, etc.) must exactly match the architecture used during training.

2. **YAMNet version**: Make sure to use the same YAMNet version from TensorFlow Hub when loading the model.

3. **Sample rate**: The model expects audio at the configured sample rate (default: 16000 Hz).

4. **Audio duration**: Input audio must be the configured duration (default: 5 seconds).
