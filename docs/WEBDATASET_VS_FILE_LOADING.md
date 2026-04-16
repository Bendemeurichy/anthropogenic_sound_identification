# WebDataset vs File-Based Loading: Key Differences

This document explains the differences between file-based CSV loading and WebDataset streaming for COI separation training.

## Overview

**File-Based Loading** (`sampler.py` + CSV):
- Loads full metadata into memory
- Samples a balanced dataset upfront (25% COI ratio)
- Creates a fixed dataset CSV for reproducibility
- Deterministic shuffling with fixed random seed

**WebDataset Loading** (`COIWebDatasetWrapper`):
- Streams samples from compressed tar shards
- Filters and samples on-the-fly during training
- More flexible but less control over exact ratios
- Efficient for large datasets that don't fit in memory

## Detailed Comparison

### 1. COI Selection (Pure COI)

**Both methods:**
- ✅ Select only "pure" COI samples (ALL labels in `target_classes`)
- ✅ Exclude mixed samples (COI + other labels)

**Implementation:**
```python
# sampler.py:get_coi()
def _is_pure_coi(labels):
    if isinstance(labels, list):
        return len(labels) > 0 and all(label in target_class for label in labels)
    elif isinstance(labels, str):
        return labels in target_class
    return False

# webdataset_utils.py:_is_coi_sample()
# Same logic - checks if ALL labels are in target_classes
```

### 2. Background Pool Exclusion

**Both methods:**
- ✅ Exclude recordings with ANY COI label (including mixed)
- ✅ Exclude recordings with no labels (None/empty)

**Implementation:**
```python
# sampler.py:sample_non_coi()
def _has_any_coi(labels):
    if labels is None or len(labels) == 0:
        return True  # Exclude unknowns
    return any(t in labels for t in target_class)

# webdataset_utils.py:_is_valid_background()
# Same logic - returns False for mixed or empty labels
```

### 3. COI Ratio Control

**File-Based:**
- ✅ **Precise ratio control** (e.g., exactly 25% COI)
- Calculates estimated segments per file based on duration
- Samples non-COI files until reaching target ratio
- Formula: `num_non_coi = num_coi * ((1 - ratio) / ratio)`

**WebDataset:**
- ⚠️ **Approximate ratio**
- Yields ALL COI samples encountered
- Background-only samples controlled by `background_only_prob`
- Final ratio depends on data distribution in shards

**Why the difference?**
- File-based can count all samples upfront and compute exact ratios
- WebDataset streams data and doesn't know totals until after full pass
- To get exact ratios with WebDataset, you'd need to:
  1. Pre-sample the metadata and create shards from sampled CSV, OR
  2. Make two passes: first to count, second to sample (defeats streaming purpose)

### 4. Segment-Based Sampling

**File-Based:**
- ✅ Accounts for file duration when sampling
- Long files contribute more "segments" to the dataset
- Ensures balanced training data by segment count, not just file count

**WebDataset:**
- ❌ Treats each file equally
- Doesn't account for duration in sampling decision
- Shorter files have same weight as longer files

**Impact:** File-based gives more accurate class balance when audio durations vary.

### 5. Reproducibility

**File-Based:**
- ✅ Fully deterministic with `random_state=42`
- Same sampling every time
- Saves sampled dataset CSV for exact reproduction

**WebDataset:**
- ⚠️ Partially deterministic
- SNR mixing uses fixed seed (`np.random.default_rng(42)`)
- Shard shuffling may vary between runs (WebDataset behavior)
- Exact sample order depends on WebDataset internals

### 6. Processing Pipeline

**File-Based:**
```
Load all metadata → Get pure COI → Sample non-COI by ratio → 
Create CSV → Load from CSV during training
```

**WebDataset:**
```
Stream from shards → Filter pure COI/valid BG on-the-fly → 
Yield samples during training
```

## When to Use Each Approach

### Use File-Based Loading When:
- ✅ Dataset fits comfortably in memory/disk
- ✅ You need exact reproducibility
- ✅ You need precise control over COI ratio
- ✅ You want segment-based sampling for varied durations
- ✅ You're doing local development/debugging

### Use WebDataset Loading When:
- ✅ Dataset is very large (hundreds of GB+)
- ✅ You're on HPC with fast scratch storage
- ✅ You want flexible COI class selection without recreating data
- ✅ Approximate ratio (±5%) is acceptable
- ✅ You want compressed storage (FLAC in tar archives)

## Current WebDataset Behavior

The current `COIWebDatasetWrapper` implementation:

✅ **Correct:**
- Pure COI filtering (ALL labels must be in target_classes)
- Mixed sample exclusion from both pools
- Empty label exclusion from background
- On-the-fly filtering by target_classes
- Dataset filtering support

⚠️ **Approximations:**
- COI ratio is approximate (depends on shard composition)
- No segment-based sampling (treats files equally)
- Shuffling may not be fully deterministic

❌ **Not Yet Implemented:**
- Two-pass ratio control (would defeat streaming purpose)
- Duration-based sampling weights

## Recommendations

### For Maximum Compatibility:
If you want WebDataset to match file-based exactly:

1. **Pre-sample the data:**
   ```bash
   # Run sampling with sampler.py
   python scripts/sample_dataset.py --output sampled.csv
   
   # Create WebDataset from sampled CSV
   python scripts/create_webdataset.py --metadata_csv sampled.csv
   ```

2. **Use WebDataset as a file replacement:**
   - WebDataset shards contain the pre-sampled, balanced dataset
   - No filtering needed at training time
   - Ratios are baked into the shards

### For Maximum Flexibility:
If you want to experiment with different COI classes:

1. **Create shards from full metadata:**
   ```bash
   python scripts/create_webdataset.py --metadata_csv all_metadata.csv
   ```

2. **Filter at training time:**
   - Update `target_classes` in config
   - `COIWebDatasetWrapper` filters dynamically
   - Approximate ratios, flexible experimentation

## Summary

Both approaches are valid with different tradeoffs:

| Feature | File-Based | WebDataset |
|---------|------------|------------|
| COI Filtering | ✅ Exact | ✅ Exact |
| Background Exclusion | ✅ Exact | ✅ Exact |
| Ratio Control | ✅ Precise | ⚠️ Approximate |
| Segment Weighting | ✅ Yes | ❌ No |
| Reproducibility | ✅ Full | ⚠️ Partial |
| Flexibility | ❌ Fixed | ✅ Dynamic |
| Memory Usage | ⚠️ Higher | ✅ Lower |
| HPC Performance | ⚠️ Slower | ✅ Faster |

Choose based on your priorities: **precision** (file-based) vs **flexibility/efficiency** (WebDataset).
