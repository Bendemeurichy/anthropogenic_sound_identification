"""
COMPREHENSIVE DIAGNOSTIC REPORT
===============================

ISSUE SUMMARY:
--------------
1. ❌ CRITICAL: Prompts are nearly identical (cosine similarity = 0.9977)
2. ❌ CRITICAL: Model outputs 98.56% energy to background, <0.01% to COI classes
3. ❌ CRITICAL: Background output is 99.99% correlated with input mixture
4. ✅ CONFIRMED: STFT/iSTFT upsampling is working perfectly (138 dB SNR)

DETAILED ANALYSIS:
------------------

1. Prompt Divergence Analysis (from checkpoint inspection):
   - airplane <-> birds:  cosine similarity = 0.9977 (❌ TOO SIMILAR)
   - airplane <-> sfx:    cosine similarity = 0.9926 (❌ TOO SIMILAR)
   - birds <-> sfx:       cosine similarity = 0.9951 (❌ TOO SIMILAR)
   - background <-> sfxbg: cosine similarity = 0.9984 (❌ TOO SIMILAR)
   
   Expected: < 0.70 for good separation
   Actual: > 0.99 (prompts are nearly identical!)

2. Separation Output Analysis:
   - Mixture energy:    100.00%
   - Plane output:        0.00% (almost silent)
   - Bird output:         0.00% (almost silent)
   - Background output:  98.56% (essentially the entire mixture)
   
   Background-mixture correlation: 0.9999 (model just copies input to background)

3. Upsampling Verification:
   - STFT → iSTFT reconstruction: 138 dB SNR (PERFECT)
   - No length mismatch
   - Conclusion: Upsampling is NOT the problem

ROOT CAUSE:
-----------
The prompts did NOT diverge during training because:

1. **Variable Prompts with High Dropout**: Config shows:
   ```yaml
   variable_prompts: true
   prompt_dropout_prob: 0.5
   min_coi_prompts: 0
   ```
   
   This means during training, each COI prompt is randomly dropped 50% of the time.
   When a prompt is dropped frequently, it doesn't receive enough gradient updates
   to diverge from its initialization.

2. **Insufficient Initialization Noise**: Train.py line 2029 shows:
   ```python
   noise = torch.randn_like(init_val) * 0.15
   ```
   
   0.15 stddev noise is not enough when combined with variable prompts.
   The prompts start too close together and variable prompts prevent them
   from diverging.

3. **Same Initialization Source**: Both airplane and birds initialize from 'sfx'
   prompt, so they start nearly identical and never separate due to variable
   prompts preventing consistent gradient updates.

RECOMMENDED FIXES:
------------------

Option A: Disable Variable Prompts (RECOMMENDED for 2-class COI)
   1. Set `variable_prompts: false` in training_config.yaml
   2. This ensures both prompts are ALWAYS present during training
   3. They'll receive consistent gradients and can diverge properly
   
Option B: Keep Variable Prompts BUT increase divergence
   1. Increase initialization noise from 0.15 to 0.30-0.50
   2. Decrease prompt_dropout_prob from 0.5 to 0.2
   3. Increase min_coi_prompts to at least 1 (keep at least one COI active)
   4. Initialize each prompt from DIFFERENT sources (not both from 'sfx')

Option C: Train with fixed prompts first, then enable variable prompts
   1. Train for 10-20 epochs with variable_prompts=false
   2. Let prompts diverge and learn distinct representations
   3. Then enable variable_prompts for robustness training

RECOMMENDED IMMEDIATE ACTION:
-----------------------------
1. Retrain from scratch with `variable_prompts: false`
2. Monitor prompt divergence every 5 epochs using inspect_checkpoint.py -d
3. Target cosine similarity < 0.70 between airplane and birds prompts
4. Training should show COI energy increasing (not just background)

FILE TO MODIFY: training_config.yaml
Change:
  variable_prompts: true    →  variable_prompts: false

This single change should fix the separation issue!
"""

print(__doc__)
