# Pre-Testing Checklist

Before running your first training session with the new extending functionality, verify:

## ✅ Code Validation

- [x] **Syntax check passed**: `train.py` compiles without errors
- [x] **All TODOs removed**: No pending implementation tasks
- [x] **Optimizer state fix applied**: Handles extending mode correctly
- [x] **Parameter groups implemented**: new/continuing/frozen prompts
- [x] **Differential LRs configured**: `existing_prompt_lr_multiplier: 0.1`

## ✅ Documentation Complete

- [x] **FINAL_SUMMARY.md**: Complete implementation overview
- [x] **EXTENDING_PROMPTS.md**: Detailed user guide
- [x] **EXAMPLE_EXTENDING_CHECKPOINT.md**: Practical walkthrough
- [x] **OPTIMIZER_STATE_FIX.md**: Critical fix documentation
- [x] **DEVICE_HANDLING.md**: GPU and checkpoint loading guide
- [x] **QUICK_REFERENCE.md**: Command reference
- [x] **IMPLEMENTATION_SUMMARY.md**: Original planning doc

## ✅ Tools Available

- [x] **inspect_checkpoint.py**: CLI tool for inspecting checkpoints
  ```bash
  python inspect_checkpoint.py checkpoints/tuss/best_model.pt
  ```

## 📋 Before First Test Run

### 1. Environment Setup
```bash
# Activate virtual environment
source /home/bendm/Thesis/project/code/.venv/bin/activate

# Verify working directory
cd /home/bendm/Thesis/project/code/src/models/tuss

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. Inspect Existing Checkpoint
```bash
# See what prompts exist
python inspect_checkpoint.py checkpoints/tuss/best_model.pt

# Compare with config
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -c training_config.yaml
```

Expected output should show:
- COI prompt: `airplane`
- Background: `background`
- Base prompts: `sfx`, `sfxbg`, `bass`, `drums`, `vocals`, `other`, `speech`, `musicbg`

### 3. Backup Your Checkpoint
```bash
# Create backup before testing
cp checkpoints/tuss/best_model.pt checkpoints/tuss/best_model_backup.pt
```

### 4. Review Configuration
```bash
# Check current config
cat training_config.yaml | grep -A5 "coi_prompts\|resume_from\|lr:"
```

Verify:
- `coi_prompts` is correct
- `resume_from` path is correct (or commented out for fresh start)
- `lr` and `existing_prompt_lr_multiplier` are reasonable

## 🧪 Suggested Test Sequence

### Test 1: Continue Fine-tuning (Safest)
```yaml
# training_config.yaml
model:
  coi_prompts: ["airplane"]

training:
  resume_from: "checkpoints/tuss/best_model.pt"
  num_epochs: 2  # Short test run
```

**Expected behavior**:
- ✓ Loads optimizer/scheduler state
- ✓ Airplane trains at reduced LR (1e-5)
- ✓ Base prompts frozen
- ✓ No new prompts

**Run**:
```bash
python train.py
```

**Look for**:
```
🔄 Continuing prompts (reduced LR: 1.0e-05):
   airplane, background
❄️  Frozen prompts (no training):
   sfx, sfxbg, bass, drums, vocals, other, speech, musicbg
```

### Test 2: Extend with New Prompt (Main Feature)

**Prerequisites**:
1. Prepare dataset with new class (e.g., "bird")
2. Update data loading paths if needed

```yaml
# training_config.yaml
model:
  coi_prompts: ["airplane", "bird"]

training:
  resume_from: "checkpoints/tuss/best_model.pt"
  num_epochs: 2  # Short test run
```

**Expected behavior**:
- ⚠️ Skips optimizer/scheduler state
- ✓ Bird trains at full LR (1e-4)
- ✓ Airplane trains at reduced LR (1e-5)
- ✓ Base prompts frozen

**Run**:
```bash
python train.py
```

**Look for**:
```
⚠️  Model is being extended with new/frozen prompts
   Skipping optimizer state loading (will start fresh)

🆕 New prompts (full LR: 1.0e-04):
   bird
🔄 Continuing prompts (reduced LR: 1.0e-05):
   airplane, background
❄️  Frozen prompts (no training):
   sfx, sfxbg, bass, drums, vocals, other, speech, musicbg
```

### Test 3: Fresh Start (Verification)

```yaml
# training_config.yaml
model:
  coi_prompts: ["train"]

training:
  # resume_from commented out
  num_epochs: 2
```

**Expected behavior**:
- ✓ All prompts are new
- ✓ No frozen prompts
- ✓ Fresh optimizer/scheduler

## ⚠️ Common Issues & Solutions

### Issue: "Checkpoint file not found"
```bash
# Verify path
ls -lh checkpoints/tuss/best_model.pt

# If not found, check actual location
find . -name "*.pt" -type f
```

### Issue: "CUDA out of memory"
```yaml
# Reduce in training_config.yaml:
training:
  batch_size: 4  # Reduce from 8
  max_clip_duration: 6.0  # Reduce from 12.0
  grad_accum_steps: 4  # Increase from 2
```

### Issue: "Prompt not found in model"
```bash
# Inspect checkpoint to see available prompts
python inspect_checkpoint.py checkpoints/tuss/best_model.pt -v
```

### Issue: "Model not improving"
- Check dataset has examples for all classes
- Verify data loading with `--verbose` flag
- Increase `num_epochs`
- Check validation metrics for overfitting

## 📊 What to Monitor

### During Training
1. **Parameter group assignment**: Verify prompts in correct groups
2. **Learning rates**: Check initial LRs are correct
3. **Loss curves**: Training should decrease
4. **Validation metrics**: Per-class SI-SNR
5. **GPU memory**: Should be stable

### After Training
1. **Checkpoint saved**: `checkpoints/tuss/best_model.pt`
2. **Training history**: `checkpoints/tuss/training_history.json`
3. **Validation loss**: Should improve or stay stable
4. **No degradation**: Existing classes maintain performance

## 🎯 Success Criteria

After test runs, the system should demonstrate:

- [ ] **Continue mode works**: Airplane continues training at reduced LR
- [ ] **Extend mode works**: New class added with differential LRs
- [ ] **Frozen prompts stay frozen**: `requires_grad=False` maintained
- [ ] **No crashes**: Training completes without errors
- [ ] **Checkpoints valid**: Can be loaded and inspected
- [ ] **Metrics make sense**: Loss decreases, SI-SNR improves
- [ ] **No degradation**: Existing classes maintain quality

## 📝 Log Key Observations

Keep track of:
1. Which test configurations worked
2. Any error messages or warnings
3. Training speed and memory usage
4. Quality of separated audio (subjective)
5. Any unexpected behavior

## 🚀 Ready to Test?

If all checklist items are complete:
1. Backup your checkpoint
2. Start with Test 1 (safest)
3. Monitor output carefully
4. Proceed to Test 2 if Test 1 succeeds
5. Document any issues

Good luck! 🎉

---

**Implementation Date**: March 27, 2026  
**Status**: Ready for user testing  
**Documentation**: Complete  
**Code Review**: Passed
