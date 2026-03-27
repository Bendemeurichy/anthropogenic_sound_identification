# Optimizer State Loading Fix

## Problem

When **extending** a model by adding new prompts, the optimizer parameter groups change:
- **Old checkpoint**: 1-2 param groups (e.g., just prompts, or prompts + backbone)
- **New optimizer**: 3-4 param groups (new_prompts, continuing_prompts, frozen_prompts, backbone)

PyTorch's `optimizer.load_state_dict()` requires the **exact same number of parameter groups** and will fail with:
```
RuntimeError: loaded state dict has a different number of parameter groups
```

## Solution

The training code now detects when the model is being **extended** vs **continued**:

### Continue Fine-tuning Mode
- All config prompts exist in checkpoint
- No new prompts, no frozen prompts
- ✅ **Load optimizer state** (preserves Adam momentum)
- ✅ **Load scheduler state** (continues LR schedule)

### Extend Mode
- New prompts added OR prompts frozen
- ❌ **Skip optimizer state** (start fresh - safe for Adam)
- ❌ **Skip scheduler state** (restart with warmup)

### Implementation

```python
# train.py:1873-1919
has_new_prompts = bool(param_groups['new_prompts'])
has_frozen_prompts = bool(param_groups['frozen_prompts'])
is_extending = has_new_prompts or has_frozen_prompts

if is_extending:
    print("  ⚠️  Model is being extended with new/frozen prompts")
    print("     Skipping optimizer state loading (will start fresh)")
else:
    # Continue fine-tuning: load optimizer state
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

## Why This is Safe

### For Continuing Prompts
- Starting with fresh Adam momentum is **safe and common practice**
- The model weights are loaded correctly (which is most important)
- Adam will quickly build new momentum/variance estimates in first few iterations
- Differential learning rates are still applied correctly

### For New Prompts
- New prompts have no optimizer state in checkpoint anyway
- Fresh optimizer state is required for them

### Training Stability
- **Warmup** helps mitigate any instability from fresh optimizer state
- First few epochs allow Adam to calibrate momentum
- This is standard when fine-tuning with different learning rates

## Testing

The fix was validated with:

### Test 1: Continue Fine-tuning
```python
# Same param groups → load succeeds
old_optimizer = AdamW([{'params': [p1], 'lr': 0.001}])
new_optimizer = AdamW([{'params': [p1], 'lr': 0.0001}])
new_optimizer.load_state_dict(old_optimizer.state_dict())  # ✓ Works
```

### Test 2: Extending
```python
# Different param groups → would fail, so we skip
old_optimizer = AdamW([{'params': [p1]}])
new_optimizer = AdamW([
    {'params': [p1], 'name': 'continuing'},
    {'params': [p2], 'name': 'new'}
])
# Skip loading old state → ✓ No error, fresh momentum
```

## User Experience

### Extending Model Output
```
Resuming training state from checkpoint: checkpoints/tuss/best_model.pt
⚠️  Model is being extended with new/frozen prompts
   Skipping optimizer state loading (will start fresh)
   This is expected and safe - Adam will build new momentum for all parameters
⚠️  Starting fresh scheduler (will apply warmup from beginning)
Resumed at epoch 24, global_step 1200, best_val_loss -8.21, lr 1.0e-04
```

### Continuing Model Output
```
Resuming training state from checkpoint: checkpoints/tuss/best_model.pt
✓ Loaded optimizer state
✓ Loaded scheduler state
Resumed at epoch 24, global_step 1200, best_val_loss -8.21, lr 5.2e-05
```

## Related Files

- `train.py:1865-1921` - Optimizer/scheduler state loading logic
- `train.py:1292-1520` - `create_model()` returns param_groups
- `train.py:1783-1819` - Multi-group optimizer creation

## Date

Fixed: 2026-03-27
