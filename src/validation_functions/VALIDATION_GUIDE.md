# Validation Pipeline: Bird vs Airplane Experiments

## Overview

The validation pipeline supports both **airplane** and **bird** classification experiments using the **same dataset CSV** with automatic COI (Class of Interest) synonym switching.

## Dataset Statistics

### Full Test Set (6,051 samples)
- **Birds**: 1,999 samples
- **Airplanes**: 156 samples  
- **Background**: 2,582 samples
- **Risoux (independent)**: 1,314 samples

### Non-Risoux Test Set (4,737 samples)
Used for synthetic mixture experiments:
- **Birds**: 1,999 samples
- **Airplanes**: 156 samples
- **Background**: 2,582 samples

## Key Mechanism: COI Synonym Re-Binarization + Class Balancing

Both `test_pipeline.py` and `test_noise_increase.py` automatically **re-binarize labels** based on the classifier type:

### For Airplane Experiments
```python
classifier_type = "plane"  # or "ast_finetuned" or "pann_finetuned"
# → pipeline.coi_synonyms = AIRPLANE_SYNONYMS
# → Label re-binarization:
#    - 156 airplane samples → label=1 (COI)
#    - 1,999 bird + 2,582 background → label=0 (non-COI)
# → Class balancing (default: enabled):
#    - Before: 156 COI, 4,581 background (ratio 1:29)
#    - After: 156 COI, 156 background (ratio 1:1)
```

### For Bird Experiments
```python
classifier_type = "bird_mae"  # or "audioprotopnet"
# → pipeline.coi_synonyms = BIRD_SYNONYMS
# → Label re-binarization:
#    - 1,999 bird samples → label=1 (COI)
#    - 156 airplane + 2,582 background → label=0 (non-COI)
# → Class balancing (default: enabled):
#    - Before: 1,999 COI, 2,738 background (ratio 1:1.37)
#    - After: No balancing needed (ratio < 2.0)
```

### Why Class Balancing?

**Enabled by default** for fair confusion matrix visualization and metrics:
- Airplane validation: 1:29 imbalance → balanced to 1:1 (downsample background)
- Bird validation: 1:1.37 imbalance → no balancing needed (close to balanced)
- Only applies when ratio > 2.0
- Disable with `balance_classes=False` if you want to evaluate on the full imbalanced set

## Implementation Details

### 1. Auto-Detection (test_pipeline.py:845-854)
```python
if self.coi_synonyms is None:
    self.coi_synonyms = get_coi_synonyms_for_classifier(classifier_type)
    # Returns AIRPLANE_SYNONYMS or BIRD_SYNONYMS based on classifier_type
```

### 2. Label Re-Binarization (test_pipeline.py:2189-2197)
```python
# Use orig_label (raw string label from dataset) if available
raw_series = df_split["orig_label"] if "orig_label" in df_split.columns else df_split["label"]
# Re-binarize using COI synonyms
df_split["label"] = raw_series.apply(
    lambda x: 1 if _is_coi_label(x, self.coi_synonyms) else 0
)
```

### 3. COI/Background Extraction (test_noise_increase.py:606-609)
```python
coi_syns = getattr(pipeline, 'coi_synonyms', None)
df_coi = _extract_coi_df(df, coi_synonyms=coi_syns)
df_bg = _extract_bg_df(df, coi_synonyms=coi_syns)
```

## Validation Test Stages

### Standard Test Sets (4,737 non-Risoux samples)
Runs 4 test stages with synthetic mixtures:
1. **Clean COI — Classification Only**: Baseline classifier performance on unmixed COI samples
2. **Clean COI — Separation + Classification**: Tests if separator preserves COI when no interference present
3. **Synthetic Mixtures — Classification Only**: Baseline classifier robustness to interference
4. **Synthetic Mixtures — Separation + Classification**: Tests separator effectiveness at removing interference

### Independent Test Sets (Risoux: 1,314 samples)
Runs 2 test stages **as-is** (no synthetic mixtures):
1. **As-is Audio — Classification Only**: Baseline classifier on real field recordings
2. **As-is Audio — Separation + Classification**: Separator effectiveness on real field recordings

## Running Validations

### Airplane Validation
```bash
cd src/validation_functions
python run_airplane_validation.py
```

**Configuration:**
- `CLASSIFIER_TYPE = "plane"` (or "ast_finetuned", "pann_finetuned")
- `SEP_CHECKPOINT`: Path to airplane-trained TUSS checkpoint
- `DATA_CSV`: CSV with orig_label column (from TUSS training)
- `EXCLUDE_DATASETS = ["risoux_test"]`: Exclude Risoux from synthetic mixtures

**Expected Sample Counts:**
- COI (airplanes): 156 samples
- Background (before balancing): 4,581 samples (1,999 birds + 2,582 background)
- Background (after balancing): 156 samples (downsampled)
- Risoux (evaluated separately): 1,314 samples

### Bird Validation
```bash
cd src/validation_functions
python run_bird_validation.py
```

**Configuration:**
- `CLASSIFIER_TYPE = "bird_mae"` (or "audioprotopnet")
- `SEP_CHECKPOINT`: Path to bird-trained TUSS checkpoint
- `DATA_CSV`: CSV with orig_label column (from TUSS training)
- `EXCLUDE_DATASETS = ["risoux_test"]`: Exclude Risoux from synthetic mixtures

**Expected Sample Counts:**
- COI (birds): 1,999 samples
- Background (before balancing): 2,738 samples (156 airplanes + 2,582 background)
- Background (after balancing): 2,738 samples (no balancing - ratio 1:1.37 < 2.0)
- Risoux (evaluated separately): 1,314 samples

## Important Notes

1. **CSV Requirements**: The dataset CSV must have an `orig_label` column that preserves the original string labels (e.g., "bird", "airplane", "background"). This is automatically created by the TUSS training script (train.py:3010-3058).

2. **Risoux Handling**: Risoux is always excluded from synthetic mixture experiments and evaluated separately as an independent test set to avoid "mixture-of-mixtures" evaluation.

3. **Contamination Filtering**: Background samples that contain COI-related keywords in their orig_label are automatically filtered out to prevent false positives.

4. **Separator-Classifier Mismatch Warning**: The pipeline warns if you use a bird classifier with an airplane-trained separator (or vice versa), as the separator may not effectively separate the classifier's COI type.

5. **test_noise_increase.py COI Selection**: The noise robustness experiment (test_noise_increase.py) doesn't use a classifier but still needs `classifier_type` to determine which samples are COI vs background. For TUSS models, it auto-detects the correct `classifier_type` from `TUSS_COI_PROMPT` ("bird" → "bird_mae", "airplane" → "plane").

## Verifying Correct Behavior

### Check COI Synonyms
Both scripts print the COI synonym configuration:
```
COI SYNONYM CONFIGURATION
==================================================
Using COI synonyms: ['airplane', 'aeroplane', 'aircraft', 'plane']

Label re-binarization:
  - Airplane samples (in orig_label) → label=1 (COI)
  - Bird samples (in orig_label) → label=0 (background)
  - Other samples → label=0 (background)
==================================================
```

### Check Sample Counts
Both scripts print dataset statistics:

**Airplane validation (with balancing):**
```
CLASS BALANCING
==================================================
  Before: 156 COI, 4581 background (ratio 1:29.4)
  After:  156 COI, 156 background (ratio 1:1)
  ✓ Downsampled background from 4581 to 156 samples
==================================================

Validation on TEST set: 156 COI, 156 background
```

**Bird validation (no balancing needed):**
```
[Info] Classes are reasonably balanced (ratio 1:1.4) - no downsampling needed

Validation on TEST set: 1999 COI, 2738 background
```

## Troubleshooting

### Wrong Sample Counts?
- **Problem**: COI count doesn't match expected (156 for airplanes, 1999 for birds)
- **Solution**: Check that `classifier_type` is set correctly and that CSV has `orig_label` column

### Separator Mismatch Warning?
- **Problem**: Warning about separator trained for different COI type
- **Solution**: Use airplane-trained separator with airplane classifier, bird-trained separator with bird classifier

### Missing orig_label Column?
- **Problem**: Label re-binarization not working correctly
- **Solution**: Re-run TUSS training to generate CSV with orig_label column (or add it manually)

## Summary

✅ **Both validation scripts correctly handle airplane and bird experiments**  
✅ **Automatic COI synonym switching based on classifier_type**  
✅ **Automatic class balancing for fair confusion matrix visualization**  
✅ **Same CSV can be used for both airplane and bird validations**  
✅ **Risoux evaluated separately as independent test set**  
✅ **Expected sample counts verified for non-Risoux test set**

### Class Balancing Details
- **Enabled by default** via `balance_classes=True`
- Downsamples majority class when ratio > 2.0
- Airplane validation: 156:4,581 → 156:156 (balanced)
- Bird validation: 1,999:2,738 → no change (already balanced)
- Balancing stats saved in JSON output under `class_balancing` key
