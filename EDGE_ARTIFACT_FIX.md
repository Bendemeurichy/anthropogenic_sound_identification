# Edge Artifact Fix for TUSS Resampling

## Problem

Kaiser windowed sinc interpolation produces visible edge artifacts in spectrograms when resampling short audio fragments. These artifacts appear as bright vertical lines at fragment boundaries (visible in your `separation_output/` spectrograms).

**Why it matters:**
- You can't hear the difference, but models working in the spectrogram domain can see it
- TUSS processes audio in segments/fragments
- Edge artifacts appear at EVERY fragment boundary
- These artifacts can confuse separation models and degrade performance

## Solution Applied

### 1. Added `resample_with_padding()` function
**Location:** `src/common/audio_utils.py`

This function:
- Adds reflection padding (128 samples) before resampling
- Resamples with Kaiser window (maintaining high quality)
- Trims padding from output
- Result: Clean edges with no artifacts

### 2. Updated TUSS Inference
**Location:** `src/models/tuss/inference.py`

**Before:**
```python
resampler = create_high_quality_resampler(sr, self.sample_rate)
waveform = resampler(waveform)
```

**After:**
```python
waveform = resample_with_padding(waveform, sr, self.sample_rate)
```

### 3. Updated TUSS Training
**Location:** `src/models/tuss/train.py` (line 1009)

**Before:**
```python
waveform = self._resampler_cache.resample(waveform, sr, self.sample_rate)
```

**After:**
```python
waveform = self._resampler_cache.resample_padded(waveform, sr, self.sample_rate)
```

## Usage

### For inference:
```python
from common.audio_utils import resample_with_padding

# Resample with clean edges (no artifacts)
waveform = resample_with_padding(waveform, orig_sr=32000, target_sr=48000)
```

### With ResamplerCache:
```python
from common.audio_utils import ResamplerCache

cache = ResamplerCache()
waveform = cache.resample_padded(waveform, orig_sr=32000, target_sr=48000)
```

### Padding modes:
- `"reflect"` (default): Mirrors signal at boundaries - **best for most audio**
- `"replicate"`: Extends edge values - good for DC-offset signals
- `"constant"`: Zero padding - not recommended (causes discontinuities)

## Verification

Run these test scripts to verify the fix:
```bash
# Test basic functionality
python test_tuss_padded_resampling.py

# Visual comparison (generates spectrogram)
python visualize_tuss_edge_fix.py

# Check edge artifacts (generates comparison plot)
python test_edge_artifacts.py
```

## Benefits

1. **Cleaner spectrograms** → Better model predictions
2. **No discontinuities** at fragment boundaries
3. **More consistent** feature extraction
4. **Reduced artifacts** that confuse models
5. **Same audio quality** - just cleaner edges

## Impact

- ✅ Training: Models learn on cleaner spectrograms
- ✅ Inference: Separation output has no edge artifacts
- ✅ Quality: No audible difference, but visible improvement in spectrograms
- ✅ Performance: Minimal overhead (~128 samples padding per fragment)

## When to Use Standard Resampling

The padded version is recommended for:
- ✅ Fragment-based processing (TUSS, separation models)
- ✅ Spectrogram-based models
- ✅ Any model that analyzes spectrograms

Standard resampling is fine for:
- Time-domain-only processing
- Full-file resampling (not fragments)
- Cases where edges don't matter

## Files Modified

1. `src/common/audio_utils.py`
   - Added `resample_with_padding()` function
   - Added `ResamplerCache.resample_padded()` method

2. `src/models/tuss/inference.py`
   - Line 46: Import changed
   - Line 431-432: Uses `resample_with_padding()`

3. `src/models/tuss/train.py`
   - Line 1009-1013: Uses `resample_padded()`

## Visual Evidence

See generated comparison images:
- `test_separation_outputs/edge_artifacts_comparison.png`
- `test_separation_outputs/tuss_edge_fix_comparison.png`

The middle panels show the OLD method with visible edge artifacts (bright vertical lines).
The bottom panels show the NEW method with clean edges.
