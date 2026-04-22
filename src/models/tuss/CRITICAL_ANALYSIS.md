# CRITICAL ANALYSIS: TUSS Multi-Class Training & Inference

**Date:** 2026-04-22  
**Model:** TUSS fine-tuned for 2-class COI separation (airplanes, birds)  
**Configuration:** `training_config.yaml` with 2 COI prompts + 1 background prompt

---

## Executive Summary

🔴 **CRITICAL FINDING**: The training implementation will correctly learn 2 separate COI classes, BUT there are **significant architectural constraints** when using different prompt configurations at inference time compared to training.

---

## 1. Training Behavior Analysis

### ✅ CORRECT: Model Will Train on Both Classes

**Evidence from `train.py:1944-1952`:**
```python
n_coi = len(config.model.coi_prompts)  # n_coi = 2
n_src = n_coi + 1  # n_src = 3 (airplane, birds, background)

all_prompts = config.model.coi_prompts + [config.model.bg_prompt]
# all_prompts = ["airplane", "birds", "background"]

prompts_template = [list(all_prompts)] * _MAX_BATCH
# Every forward pass uses ALL 3 prompts
```

**Forward Pass (`train.py:1070-1073`):**
```python
prompts = prompts_template[:B]
outputs = model(mixture, prompts)
# outputs.shape = (B, 3, T) - always 3 sources
```

**Conclusion:** ✅ Training ALWAYS uses all 3 prompts simultaneously. The model learns to separate:
- COI class 0 (airplane) → output channel 0
- COI class 1 (birds) → output channel 1  
- Background → output channel 2

---

## 2. Architecture Flexibility Analysis

### ✅ TUSS Architecture Supports Variable n_src

**Evidence from `base/nets/tuss.py:137-159`:**
```python
def forward(self, input: torch.Tensor, prompts: List[str]) -> torch.Tensor:
    n_batch, n_frames, n_freqs = batch.shape[:3]
    n_src = len(prompts[0])  # <--- DYNAMIC! Determined at runtime
    
    # Process all prompts together
    batch = self._concatenate_prompt(batch, prompts)
    for block in self.cross_prompt_module:
        batch = block(batch)  # All prompts interact here
        
    # Reshape to process n_src sources independently
    batch = batch.reshape(n_batch * n_src, -1, n_frames, self.num_bands)
    
    # Output shape: (n_batch, n_src, n_frames, n_freqs)
```

**BandSplitModule Initialization (`base/nets/tuss.py:117-118`):**
```python
self.band_split_module = BandSplitModule(
    1,  # num_src=1 per decoder forward pass
    emb_dim, stft_size, sample_rate,
)
```

**Key Insight:** The decoder is initialized with `num_src=1`, but handles multiple sources by batching them as `(B*n_src, ...)`. This makes the architecture **flexible** with respect to the number of sources at inference time.

**Conclusion:** ✅ The architecture CAN handle variable numbers of prompts at inference (e.g., 1 COI + BG, 2 COI + BG, etc.)

---

## 3. 🔴 CRITICAL ISSUE: Cross-Prompt Dependencies

### Problem: Training Creates Inter-Prompt Dependencies

**Evidence from `base/nets/tuss.py:172-174`:**
```python
# ALL prompts are concatenated and processed together
batch = self._concatenate_prompt(batch, prompts)  # Shape: (B, C, T+n_src, F)
for block in self.cross_prompt_module:
    batch = block(batch)  # <--- Cross-attention across ALL prompts!
```

**What This Means:**
- During training with `["airplane", "birds", "background"]`, the cross-prompt module learns attention patterns that involve ALL THREE prompts
- The "airplane" prompt learns to attend to both "birds" and "background"
- The "birds" prompt learns to attend to both "airplane" and "background"
- These prompts are **interdependent** through the attention mechanism

### 🔴 Impact on Inference with Fewer Prompts

**Scenario 1: Inference with only `["airplane", "background"]` (2 prompts)**
- ❌ Cross-prompt attention patterns are different (2 prompts instead of 3)
- ❌ "airplane" prompt never learned to work WITHOUT "birds" prompt present
- ❌ Potential performance degradation compared to training configuration

**Scenario 2: Inference with only `["birds", "background"]` (2 prompts)**
- ❌ Same issue - "birds" prompt never learned to work WITHOUT "airplane" prompt
- ❌ Attention patterns are structurally different from training

**Scenario 3: Inference with all `["airplane", "birds", "background"]` (3 prompts)**
- ✅ Matches training configuration exactly
- ✅ Expected to work optimally

---

## 4. Loss Function Analysis

### ✅ Loss Function Handles Multi-Class Correctly

**Evidence from `train.py:441-454`:**
```python
class COIWeightedSNRLoss:
    def __init__(self, n_src, ...):
        self.n_src = n_src  # 3 (airplane, birds, background)
        self.n_coi = n_src - 1  # 2 (airplane, birds)
        
    def forward(self, est, ref):
        # Shape: est=(B, 3, T), ref=(B, 3, T)
        
        # Active-class weighting: only average over ACTIVE COIs
        ref_power = (ref[:, :self.n_coi] ** 2).mean(dim=-1)  # (B, 2)
        is_active = ref_power > SILENCE_ENERGY_EPS  # (B, 2)
        
        coi_losses = per_src[:, :self.n_coi]  # (B, 2)
        active_count = is_active.sum(dim=-1).clamp(min=1)
        coi_loss = (coi_losses * is_active).sum(dim=-1) / active_count
        
        bg_loss = per_src[:, -1]  # (B,)
        weighted = (self.coi_weight * coi_loss + bg_loss) / (self.coi_weight + 1.0)
```

**Behavior:**
- ✅ When only airplane is present: `is_active = [True, False]` → only airplane loss counted
- ✅ When only birds is present: `is_active = [False, True]` → only birds loss counted
- ✅ When both are present: `is_active = [True, True]` → both losses averaged
- ✅ When neither is present (background-only): `active_count=1` prevents division by zero

**Conclusion:** ✅ Loss function correctly handles all possible class presence combinations during training.

---

## 5. Dataset Handling Analysis

### ✅ Dataset Creates Correct Multi-Class Targets

**Evidence from `train.py:856-884`:**
```python
# Single COI example (e.g., airplane):
sources = [torch.zeros(...) for _ in range(self.n_coi_classes)]  # [0, 0]
sources[class_idx] = coi_audio  # class_idx=0 for airplane
# Result: sources = [airplane_audio, silence]

# Multi-COI example (30% probability):
if self._rng.random() < self.multi_coi_prob:  # 0.3
    # Add second COI from different class
    sources[class_idx_2] = coi_audio_2
    # Result: sources = [airplane_audio, bird_audio]

# Append background
sources.append(background)
# Final: [airplane, bird, background] or [airplane, 0, background], etc.
```

**Class Distribution Tracking (`train.py:1105-1112`):**
```python
ref_power = (clean_wavs[:, :criterion.n_coi] ** 2).mean(dim=-1)
is_active = ref_power > SILENCE_ENERGY_EPS
active_class_counts = is_active.sum(dim=0).cpu()  # (2,)
# Logs: c0=16 c1=16 (e.g., 16 airplane samples, 16 bird samples per batch)
```

**Conclusion:** ✅ Dataset correctly creates multi-class targets where each sample has:
- 0, 1, or 2 active COI channels (most common: 1 active)
- Always 1 active background channel
- Proper class indexing: airplane=channel 0, birds=channel 1

---

## 6. Validation Reporting

### ✅ Per-Class Metrics Reported Correctly

**Evidence from `train.py:1274-1320`:**
```python
for cls_i in range(n_src - 1):  # For each COI class
    ref_energy = clean_wavs[:, cls_i].pow(2).mean(dim=-1)
    present = ref_energy > SILENCE_ENERGY_EPS
    if present.any():
        snr_val = -si_snr_loss(outputs[:, cls_i:cls_i+1], clean_wavs[:, cls_i:cls_i+1])
        per_class_sisnr[cls_i].append(float(snr_val[present].mean().item()))
        per_class_counts[cls_i] += present.sum().item()

# Output format:
# Val SI-SNR – cls0: 12.34 dB [107 samples], cls1: 15.67 dB [1648 samples], BG: 18.90 dB
```

**Conclusion:** ✅ Validation correctly computes and reports SI-SNR per class, only counting samples where each class is actually present.

---

## 7. 🔴 INFERENCE RECOMMENDATIONS

### ✅ SAFE: Always Use Full Prompt Set
```python
# At inference, ALWAYS use the same prompts as training
prompts = [["airplane", "birds", "background"]]
outputs = model(mixture, prompts)  # Shape: (1, 3, T)

# To get only airplane:
airplane_separated = outputs[0, 0, :]  # First COI channel

# To get only birds:
birds_separated = outputs[0, 1, :]  # Second COI channel

# To get background:
background = outputs[0, 2, :]  # Background channel
```

### ⚠️ RISKY: Using Subset of Prompts
```python
# This WILL work architecturally but may perform worse
prompts = [["airplane", "background"]]  # Only 2 prompts
outputs = model(mixture, prompts)  # Shape: (1, 2, T)
```

**Why Risky:**
1. Cross-prompt attention patterns differ from training (2 prompts vs 3)
2. "airplane" prompt never learned to work without "birds" prompt present
3. No empirical data on performance degradation magnitude

### 📊 RECOMMENDED: Empirical Testing
If you need single-class inference, test both approaches:
1. **Full prompts (recommended):** Use all 3 prompts, extract desired channel
2. **Subset prompts (experimental):** Use only needed prompts, measure performance drop

---

## 8. 🟢 SUMMARY: Will It Work?

### YES ✅ - Training Will Work Correctly
- Model WILL learn to separate 2 distinct COI classes (airplane, birds)
- Loss function correctly handles variable class presence
- Per-class augmentation (16x airplane, 1x birds) will balance dataset
- Validation metrics will accurately report per-class performance

### YES ✅ - Inference With All Prompts Will Work
- Using `["airplane", "birds", "background"]` at inference matches training
- Extract individual classes from output channels 0, 1, 2
- Expected optimal performance

### MAYBE ⚠️ - Inference With Subset Prompts May Degrade
- Architecture supports variable n_src
- BUT cross-prompt dependencies may hurt performance
- Magnitude of degradation unknown without testing
- Recommend always using full prompt set and selecting output channels

---

## 9. Action Items

### Immediate (No Code Changes Needed)
1. ✅ Proceed with training using current configuration
2. ✅ Use full prompt set at inference: `["airplane", "birds", "background"]`
3. ✅ Select desired output by indexing: `outputs[0, class_idx, :]`

### Optional (Empirical Validation)
1. After training, benchmark inference performance:
   - Full prompts (baseline): `["airplane", "birds", "background"]`
   - Airplane-only: `["airplane", "background"]`
   - Birds-only: `["birds", "background"]`
2. Measure SI-SNR difference to quantify degradation
3. Document findings for future reference

### Advanced (If Subset-Prompt Inference Is Critical)
1. Add "prompt dropout" during training (randomly drop COI prompts)
2. Model learns to work with variable prompt sets
3. Requires architecture modification and retraining

---

## 10. Conclusion

The TUSS training implementation is **fundamentally sound** and will correctly learn to separate 2 COI classes. The architecture is flexible enough to handle variable numbers of prompts at inference, but **optimal performance is guaranteed only when using the same prompt configuration as training**.

**Bottom Line:** Train with 2 COI classes, infer with all 3 prompts, extract desired channels. This approach is architecturally sound and will work as expected.
