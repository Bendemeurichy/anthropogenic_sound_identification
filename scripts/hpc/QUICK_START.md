# Quick Reference: Enable WebDataset on HPC

## TL;DR - What to Edit

Once you're on the HPC and know your paths (`$VSC_DATA` and `$VSC_SCRATCH`), edit these three files:

### 1. `src/models/sudormrf/training_config.yaml`
```yaml
data:
  # ... other settings ...
  use_webdataset: true                                              # Change false → true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/webdataset_shards"  # Fill in your path

# ... model settings ...

training:
  # ... other settings ...
  checkpoint_dir: "/data/gent/464/vsc46423/checkpoints/sudormrf"   # Fill in your path
```

### 2. `src/models/tuss/training_config.yaml`
```yaml
data:
  # ... other settings ...
  use_webdataset: true                                              # Change false → true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/webdataset_shards"  # Fill in your path

# ... model settings ...

training:
  # ... other settings ...
  checkpoint_dir: "/data/gent/464/vsc46423/checkpoints/tuss"       # Fill in your path
  device: "cuda"                                                    # Change "cuda:1" → "cuda"
```

### 3. `src/models/clapsep/training_config.yaml`
```yaml
data:
  # ... other settings ...
  use_webdataset: true                                              # Change false → true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/webdataset_shards"  # Fill in your path

# ... model settings ...

training:
  # ... other settings ...
  checkpoint_dir: "/data/gent/464/vsc46423/checkpoints/clapsep"    # Fill in your path
```

**Note:** The CLAPSep PBS script (`scripts/hpc/train_clapsep.pbs`) already has `--use-webdataset` and `--webdataset-path` flags that override the config file, so it will work with just the environment variables `$VSC_SCRATCH` and `$VSC_DATA`.

## Find Your Paths on HPC
```bash
echo "Replace this in configs:"
echo "  VSC_SCRATCH = $VSC_SCRATCH"
echo "  VSC_DATA    = $VSC_DATA"
```

## Submit Jobs
```bash
qsub scripts/hpc/train_sudormrf.pbs
qsub scripts/hpc/train_tuss.pbs
qsub scripts/hpc/train_clapsep.pbs
```

## That's It!
For detailed instructions, see `scripts/hpc/WEBDATASET_SETUP.md`
