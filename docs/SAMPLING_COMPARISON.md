# Sampling Comparison: File-Based vs WebDataset

## Quick Answer: ⚠️ **Almost, but Not Exactly**

The WebDataset implementation is now **very close** to file-based sampling, but there are still some key differences.

---

## Side-by-Side Comparison

| Feature | File-Based (`sampler.py`) | WebDataset (Current) | Match? |
|---------|---------------------------|----------------------|--------|
| **Pure COI Selection** | ✅ Only samples with ALL labels in target_classes | ✅ Only samples with ALL labels in target_classes | ✅ YES |
| **Mixed Sample Exclusion** | ✅ Excludes COI+other from both pools | ✅ Excludes COI+other from both pools | ✅ YES |
| **Empty Label Exclusion** | ✅ Excludes None/empty from background | ✅ Excludes None/empty from background | ✅ YES |
| **Random Seed** | ✅ `random_state=42` | ✅ `seed=42` (now passed through) | ✅ YES |
| **COI Ratio Control** | ✅ Precise 25% via segment counting | ⚠️ Approximate (yields all COI) | ❌ NO |
| **Segment-Based Sampling** | ✅ Weights by duration/segments | ❌ Treats files equally | ❌ NO |
| **Deterministic Subset** | ✅ Fixed subset saved to CSV | ⚠️ Streams all, filters on-the-fly | ⚠️ PARTIAL |
| **Background Sampling** | ✅ Samples exact count per split | ⚠️ Uses all matching backgrounds | ❌ NO |

---

## Detailed Breakdown

### 1. ✅ **Pure COI Selection** - MATCHES

**Both implementations:**
```python
# Select only samples where ALL labels are in target_classes
def _is_pure_coi(labels):
    return all(label in target_classes for label in labels)
```

**Result:** ✅ Identical behavior

---

### 2. ✅ **Mixed Sample Exclusion** - MATCHES

**Both implementations:**
```python
# Exclude samples with ANY COI label from background pool
def _has_any_coi(labels):
    return any(label in target_classes for label in labels)
```

**Result:** ✅ Identical behavior

---

### 3. ✅ **Empty Label Exclusion** - MATCHES

**Both implementations:**
- Exclude `None`, empty lists, NaN from background pool
- Only valid labeled backgrounds are used

**Result:** ✅ Identical behavior

---

### 4. ✅ **Random Seed** - NOW MATCHES

**File-Based:**
```python
shuffled_non_coi = non_coi_split_df.sample(frac=1, random_state=42)
```

**WebDataset:**
```python
self._rng = np.random.default_rng(seed)  # seed=42 from config
bg_idx = self._rng.integers(len(bg_buffer))  # Seeded sampling
```

**Result:** ✅ Both use seed=42, but different RNG implementations (pandas vs numpy)

---

### 5. ❌ **COI Ratio Control** - DOES NOT MATCH

**File-Based:**
```python
# Calculate exact segments needed for 25% COI ratio
num_coi_segments = coi_split_df["est_segments"].sum()
num_non_coi_segments_needed = int(num_coi_segments * ((1 - 0.25) / 0.25))

# Sample backgrounds until reaching target segment count
mask = cumulative_segments <= num_non_coi_segments_needed
sampled_non_coi_split_df = shuffled_non_coi.head(mask.sum() + 1)
```

**WebDataset:**
```python
# Yields EVERY COI sample encountered
if is_coi:
    yield self._create_mixture(coi_sample, bg_buffer)

# Background-only controlled by probability, not count
if self._rng.random() < self.background_only_prob:
    yield self._create_background_only(bg_buffer)
```

**Why Different?**
- File-based knows total COI count upfront → calculates exact background count
- WebDataset streams data → doesn't know totals until after full pass
- WebDataset yields all COI, backgrounds via probability

**Impact:** 
- File-based: **Exactly** 25% COI
- WebDataset: **Variable** (depends on shard composition, maybe 20-35% COI)

---

### 6. ❌ **Segment-Based Sampling** - DOES NOT MATCH

**File-Based:**
```python
def est_segments(dur):
    if dur < segment_duration:
        return 1
    return 1 + int((dur - segment_duration) / segment_stride)

# Long files contribute more to the ratio calculation
metadata_df["est_segments"] = metadata_df["duration"].apply(est_segments)
cumulative_segments = shuffled_non_coi["est_segments"].cumsum()
```

**WebDataset:**
```python
# Each file/sample treated equally, no duration weighting
# If you have:
#   - 100 short COI files (2 sec each)
#   - 10 long background files (60 sec each)
# Both treated as equal counts (100 COI, 10 background)
```

**Impact:**
- File-based: 100 COI files (2 sec) = 100 segments, needs ~300 BG segments (long files sampled accordingly)
- WebDataset: 100 COI files treated as 100 samples, 10 BG files as 10 samples → heavy COI bias

---

### 7. ⚠️ **Deterministic Subset** - PARTIAL MATCH

**File-Based:**
```python
# Creates a fixed, reproducible subset
sampled_df = sample_non_coi(metadata_df, coi_df, coi_ratio=0.25)
sampled_df.to_csv("dataset.csv")  # Saved for reproducibility

# Training always uses the SAME samples
train_loader = create_dataloader("dataset.csv")
```

**WebDataset:**
```python
# Streams ALL samples from shards, filters on-the-fly
# Each epoch may see different samples if:
#   - Shard shuffle is enabled
#   - WebDataset internal shuffling varies

# BUT: If shards are created from pre-sampled CSV, then:
#   - Shards contain the fixed subset
#   - Reproducible like file-based
```

**Result:** 
- ✅ If shards created from pre-sampled CSV → Same subset
- ⚠️ If shards created from full metadata → Different subsets each run

---

### 8. ❌ **Background Sampling Count** - DOES NOT MATCH

**File-Based:**
```python
# Samples EXACTLY the number of backgrounds needed for 25% ratio
for split in ["train", "val", "test"]:
    num_non_coi_needed = calculate_from_coi_count()
    sampled_backgrounds = shuffle_and_sample(num_non_coi_needed)
```

**WebDataset:**
```python
# Uses ALL valid backgrounds in the shards
# Only yields background-only samples with probability
# No per-split balancing
```

**Impact:**
- File-based: Each split balanced independently (train=25%, val=25%, test=25%)
- WebDataset: Ratio depends on overall shard composition

---

## Summary: What Works vs What Doesn't

### ✅ **What Works (Matches File-Based):**
1. Pure COI filtering logic
2. Mixed sample exclusion logic
3. Empty label exclusion logic
4. Seeded random sampling (with seed=42)
5. Dataset filtering support

### ⚠️ **What's Approximate:**
1. COI ratio (variable, not exactly 25%)
2. Reproducibility (depends on WebDataset internals)

### ❌ **What's Different:**
1. No segment-based weighting (duration not considered)
2. No exact background count control
3. No per-split ratio balancing

---

## Recommendations

### **Option 1: Pre-Sample Then Create Shards (Best for Exact Matching)**

Create shards from pre-sampled CSVs:

```bash
# Step 1: Run sampling to create balanced CSV
python -c "
from src.label_loading.sampler import *
from src.label_loading.metadata_loader import *

metadata = load_metadata_datasets('data', 'datasets')
sep_metadata, _ = split_seperation_classification(metadata)

target_classes = ['airplane', 'Airplane', 'plane']
coi_df = get_coi(sep_metadata, target_classes)
sampled_df = sample_non_coi(sep_metadata, coi_df, 
                             target_class=target_classes, 
                             coi_ratio=0.25)

# Add binary labels
sampled_df['label'] = sampled_df['label'].apply(
    lambda x: 1 if any(lbl in target_classes for lbl in (x if isinstance(x, list) else [x])) else 0
)

sampled_df.to_csv('sampled_dataset.csv', index=False)
print(f'Saved {len(sampled_df)} samples')
"

# Step 2: Create WebDataset shards from pre-sampled CSV
python scripts/create_webdataset.py \
    --metadata_csv sampled_dataset.csv \
    --output_dir $VSC_SCRATCH/webdataset_shards \
    --samples_per_shard 1000

# Step 3: Training uses pre-sampled shards
# Set target_classes=[] in config (already filtered in shards)
# Or set target_classes to same values (redundant filtering)
```

**Result:** ✅ Exact match with file-based

---

### **Option 2: Use Current Flexible Approach (Good Enough for Most Cases)**

Keep current WebDataset setup:

```yaml
# training_config.yaml
data:
  target_classes: ["airplane", "Airplane", "plane"]
  use_webdataset: true
  webdataset_path: "$VSC_SCRATCH/webdataset_shards"
```

**Result:** 
- ⚠️ Approximate 25% ratio (maybe 20-30%)
- ✅ Flexible COI class selection
- ✅ Good enough for most training scenarios

---

## When Does It Matter?

### **Exact Matching IS Important When:**
- 📊 Comparing results with published papers
- 🔬 Ablation studies requiring identical data
- 📈 Benchmarking different models on same data
- 🎯 Reproducibility for peer review

### **Approximate Matching IS Fine When:**
- 🚀 Initial model development
- 🔧 Hyperparameter tuning
- 🎨 Experimenting with different COI classes
- ⚡ Fast iteration on HPC

---

## Final Verdict

**Current WebDataset implementation:**
- ✅ **Filtering logic:** 100% match
- ✅ **Random seeding:** Match (with caveats)
- ⚠️ **Sampling ratio:** Approximate
- ❌ **Segment weighting:** Not implemented

**Recommendation:** 
- For production/benchmarking: Use **Option 1** (pre-sample)
- For development/flexibility: Use **Option 2** (current)

Choose based on your priority: **precision** vs **flexibility**.
