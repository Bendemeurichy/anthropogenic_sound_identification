# Device Handling in TUSS Training

## Summary

**Yes, the model DOES train on GPU** (if configured). The checkpoint loading to CPU is just best practice before moving to the target device.

## How Device Handling Works

### 1. Checkpoint Loading (CPU first)
```python
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```

**Why CPU?**
- ✅ Prevents CUDA out-of-memory errors when loading large checkpoints
- ✅ Device-agnostic: works regardless of where checkpoint was saved
- ✅ Allows explicit control of target device

### 2. Model Creation (still on CPU)
```python
model = create_model(config, resume_ckpt_path=resume_path)
```

The model architecture is built and checkpoint weights are loaded while on CPU.

### 3. Device Movement (moved to GPU)
At the end of `create_model()`:
```python
device = config.training.device  # e.g., "cuda:0"
model = model.to(device)
print(f"✓ Model successfully moved to {device}")
```

The entire model (including all parameters and buffers) is moved to GPU here.

### 4. Training Loop (GPU execution)
```python
# Data moved to GPU
sources = sources.to(device, non_blocking=True)

# Forward pass on GPU
with autocast_ctx:
    outputs = model(mixture, prompts)  # Runs on GPU

# Backward pass on GPU
loss.backward()
```

All training computations happen on GPU.

## Verifying GPU Usage

### Check Training Output

When you run `python train.py`, you should see:

```
Creating model …
Moving model to device: cuda:0
✓ Model successfully moved to cuda:0

======================================================================
TRAINING CONFIGURATION
======================================================================
Device: cuda:0
AMP enabled: True
AMP dtype: torch.bfloat16
GradScaler: False
GPU: NVIDIA RTX 4090
GPU memory: 24.00 GB
======================================================================
```

### Monitor GPU Usage

While training, check GPU utilization:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- High GPU utilization (>80%)
- Memory usage increasing during forward/backward passes
- Python process using GPU memory

### Programmatic Check

Add to your code:
```python
# After model creation
print(f"Model device: {next(model.parameters()).device}")

# During training (first batch)
print(f"Input device: {mixture.device}")
print(f"Output device: {outputs.device}")
```

## Configuration

### training_config.yaml
```yaml
training:
  device: "cuda"  # or "cuda:0", "cuda:1", etc.
  use_amp: true
  amp_dtype: "bf16"  # bfloat16 (recommended) or fp16
```

### Command-line Override
```bash
# Use default GPU (cuda:0)
python train.py --device cuda

# Use specific GPU
python train.py --device cuda:1
python train.py --gpu 1  # Shorthand

# Force CPU (for debugging)
python train.py --device cpu
```

## Device Resolution Logic

The `resolve_device()` function handles various formats:

```python
resolve_device("cuda")      # → "cuda:0" (or first available GPU)
resolve_device("cuda:2")    # → "cuda:2" (validates GPU exists)
resolve_device(1)           # → "cuda:1" (integer GPU index)
resolve_device("cpu")       # → "cpu"
```

If the requested GPU doesn't exist, it falls back to `cuda:0` or `cpu`.

## Common Issues

### Issue 1: "CUDA out of memory"

**Symptoms**: Training crashes with OOM error

**Solutions**:
- Reduce `batch_size` in config
- Reduce `segment_length` 
- Enable gradient accumulation: `grad_accum_steps: 4`
- Use `amp_dtype: "bf16"` (more memory efficient than fp32)

### Issue 2: Training on wrong GPU

**Symptoms**: GPU 0 shows activity when you specified GPU 1

**Solution**: Verify device in training output:
```
Device: cuda:1
GPU: NVIDIA RTX 4090
```

Also check:
```bash
echo $CUDA_VISIBLE_DEVICES
```

### Issue 3: Not using GPU at all

**Symptoms**: 
- nvidia-smi shows 0% utilization
- Training is very slow

**Diagnosis**:
```python
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be > 0
```

**Solutions**:
- Check CUDA installation: `python -c "import torch; print(torch.version.cuda)"`
- Reinstall PyTorch with CUDA support
- Check driver compatibility

### Issue 4: Checkpoint says "cpu" but I want GPU

**This is normal!** The checkpoint is loaded to CPU first, then moved to GPU. Check the training output:
```
Loading checkpoint: checkpoints/tuss/best_model.pt  # Loads to CPU
✓ Model successfully moved to cuda:0                # Then moved to GPU
```

## Performance Tips

### 1. Optimize Data Loading
```yaml
training:
  num_workers: 4      # Parallel data loading
  pin_memory: true    # Faster CPU→GPU transfer
```

### 2. Enable Channels Last Memory Format
For potential speedup (not currently implemented):
```python
model = model.to(device, memory_format=torch.channels_last)
```

### 3. Use Multiple GPUs (Future Enhancement)
Would require wrapping model:
```python
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

## Summary

✅ **Checkpoints load to CPU** - Best practice for flexibility  
✅ **Model moves to GPU** - Happens in `create_model()`  
✅ **Training runs on GPU** - All forward/backward passes use GPU  
✅ **Configuration works** - Device specified in config or CLI  
✅ **Verification available** - Check logs for device confirmation  

The device handling is correct and follows PyTorch best practices!
