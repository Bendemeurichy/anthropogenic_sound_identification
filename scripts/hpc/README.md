# HPC Training Setup Guide

## Overview

This directory contains scripts for training the three audio separation models on UGent HPC infrastructure.

## Files

- `setup_venv.sh` - One-time virtual environment setup (run interactively on donphan)
- `requirements_hpc.txt` - Minimal Python dependencies for training
- `train_sudormrf.pbs` - PBS job script for SudoRM-RF training
- `train_tuss.pbs` - PBS job script for TUSS training
- `train_clapsep.pbs` - PBS job script for CLAPSep training
- `optuna_tune.pbs` - PBS job script for Optuna hyperparameter tuning (all models)

## Storage Layout

| Location | Quota | Usage | Contents |
|----------|-------|-------|----------|
| `$VSC_HOME` | 3 GB | Minimal | Config files, SSH keys |
| `$VSC_DATA` | 25 GB | ~12-15 GB | Code, venv (~5-6 GB), checkpoints |
| `$VSC_SCRATCH` | 25 GB | ~18 GB | WebDataset shards (train+val only) |

## Setup Instructions

### 1. Transfer Code to HPC

```bash
# On your local machine
rsync -avP --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    ./code/ login.hpc.ugent.be:$VSC_DATA/code/
```

### 2. Transfer WebDataset Shards

First, find your actual `$VSC_SCRATCH` path:

```bash
ssh login.hpc.ugent.be
echo $VSC_SCRATCH
# Example output: /kyukon/scratch/gent/464/vsc46423/vsc46423
exit
```

Then transfer (excluding test shards to save space):

```bash
rsync -avP --exclude='test-*.tar' \
    ../data/webdataset/ \
    login.hpc.ugent.be:/kyukon/scratch/gent/464/vsc46423/vsc46423/webdataset_shards/
```

### 3. Create Virtual Environment

```bash
# SSH to HPC
ssh login.hpc.ugent.be

# Switch to donphan (fast interactive access)
module swap cluster/donphan
qsub -I

# Once on compute node
cd $VSC_DATA/code
bash scripts/hpc/setup_venv.sh

# Wait 15-25 minutes for installation
# Exit interactive job when done
exit
```

### 4. Update Training Configs

Edit the YAML config files to set paths:

**`src/models/sudormrf/training_config.yaml`:**
```yaml
data:
  use_webdataset: true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/vsc46423/webdataset_shards"

training:
  checkpoint_dir: "$VSC_DATA/checkpoints/sudormrf"
```

**`src/models/tuss/training_config.yaml`:**
```yaml
data:
  use_webdataset: true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/vsc46423/webdataset_shards"

training:
  checkpoint_dir: "$VSC_DATA/checkpoints/tuss"
```

**`src/models/clapsep/training_config.yaml`:**
```yaml
data:
  use_webdataset: true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/vsc46423/webdataset_shards"

training:
  checkpoint_dir: "$VSC_DATA/checkpoints/clapsep"
```

### 5. Submit Training Jobs

```bash
# From login node
cd $VSC_DATA/code

# Option A: Run hyperparameter tuning first (recommended)
# This finds optimal hyperparameters without saving checkpoints (saves storage)
qsub scripts/hpc/optuna_tune.pbs

# Wait for tuning to complete (~24-48 hours for 20 trials × 3 models)
# Results will be in configs/tuned/

# Then update training_config.yaml files with best hyperparameters
# and submit final training runs with full checkpoint saving

# Option B: Submit training with current configs
qsub scripts/hpc/train_sudormrf.pbs
qsub scripts/hpc/train_tuss.pbs
qsub scripts/hpc/train_clapsep.pbs

# Monitor jobs
qstat -u $USER

# Check output (job ID from qsub)
tail -f optuna_tune.o<jobid>
tail -f sudormrf_train.o<jobid>
```

## Package Optimizations

The `requirements_hpc.txt` is minimal to save storage:

### Excluded (saves ~5 GB)
- **TensorFlow** (1.8 GB) - Only for PANN classifier on separate server
- **Torchvision** (16 MB) - Not used by separation models
- **Plotly** (47 MB) - Visualization only
- **Faiss** (85 MB) - Not used in training
- **Jupyter/notebook tools** (~50 MB) - Development only

### Included (essential only)
- PyTorch + CUDA (~2 GB)
- Audio processing (librosa, soundfile, etc.)
- Model-specific (transformers, pytorch-lightning, etc.)
- Training utilities (pandas, numpy, scipy, etc.)

**Total venv size: ~5-6 GB** (down from 11 GB locally)

## Job Monitoring

```bash
# Check job status
qstat -u $USER

# Check detailed job info
qstat -f <jobid>

# View output logs
tail -f <jobname>.o<jobid>

# View error logs
tail -f <jobname>.e<jobid>

# Cancel a job
qdel <jobid>
```

## Resource Allocation

### Regular Training Jobs
Each job requests:
- **Cluster:** accelgor (A100 80GB GPUs)
- **GPUs:** 1
- **CPUs:** 8 cores
- **Memory:** 64 GB
- **Walltime:** 32 hours

### Optuna Tuning Job
- **Cluster:** accelgor (A100 80GB GPUs)
- **GPUs:** 1
- **CPUs:** 8 cores
- **Memory:** 64 GB
- **Walltime:** 48 hours (20 trials × 3 models × 5 epochs)
- **Storage:** Minimal (~few MB for Optuna DB, no model checkpoints saved)

## Hyperparameter Tuning Workflow

The recommended workflow is to run Optuna tuning first to find optimal hyperparameters, then retrain with those parameters for final model weights.

### 1. Run Optuna Tuning

Edit `scripts/hpc/optuna_tune.pbs` to set:
- `TARGET_CLASS`: The class to separate (e.g., "bird", "airplane")
- `N_TRIALS`: Number of trials per model (default: 20)
- `MAX_EPOCHS`: Epochs per trial (default: 5, optimized for abundant data)
- `CSV_PATH`: Path to your dataset CSV file

Then submit:
```bash
qsub scripts/hpc/optuna_tune.pbs
```

**Storage-Efficient Design:**
- Uses `--no-save-checkpoints` flag to avoid saving model weights during trials
- Only hyperparameters are tracked in SQLite database (~few MB)
- Saves significant HPC storage (avoids ~100+ checkpoint files)

### 2. Review Results

After completion, results are saved to:
- **Best configs**: `configs/tuned/{model}_{target}_best.yaml`
- **Study statistics**: `configs/tuned/{model}_{target}_tuning_stats.txt`
- **Optuna database**: `optuna_coi_tuning.db`

```bash
# View best hyperparameters
cat configs/tuned/tuss_bird_best.yaml

# View tuning statistics
cat configs/tuned/tuss_bird_tuning_stats.txt
```

### 3. Update Training Configs

Copy the best hyperparameters from `configs/tuned/` into your main training configs:
```bash
# Example: Update TUSS config with tuned hyperparameters
cp configs/tuned/tuss_bird_best.yaml src/models/tuss/training_config.yaml
```

Or manually update specific parameters in `src/models/{model}/training_config.yaml`.

### 4. Final Training with Best Hyperparameters

Now run full training with checkpoint saving to get final model weights:
```bash
qsub scripts/hpc/train_tuss.pbs
```

This approach ensures:
- ✓ Optimal hyperparameters for your dataset
- ✓ Minimal HPC storage usage during tuning
- ✓ Full model checkpoints only for final trained models

## Troubleshooting

### Quota Issues

Check your quota:
```bash
mmlsquota
```

Clean up if needed:
```bash
# Remove pip cache (if it exists from failed installs)
rm -rf ~/.cache/pip

# Check venv size
du -sh $VSC_DATA/venv
```

### Module Issues

If Python module isn't found:
```bash
module av Python
module spider Python/3.11.3
```

### Path Issues

Make sure you use the FULL paths in configs, not `$VSC_SCRATCH` (which doesn't expand in YAML).

## Support

For HPC issues: hpc@ugent.be
