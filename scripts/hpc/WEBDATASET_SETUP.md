# WebDataset Configuration for HPC Training

This guide shows you how to enable WebDataset mode for faster training on HPC.

## Why WebDataset?

WebDataset mode loads pre-compressed tar files from `$VSC_SCRATCH` instead of reading individual audio files. This is **much faster** on HPC because:
- Fewer file system operations (read 1 tar file vs thousands of audio files)
- Faster I/O on scratch storage
- Pre-mixed audio already prepared (no mixing overhead during training)

## Prerequisites

1. ✅ Transfer WebDataset tar files to HPC:
   ```bash
   # On your local machine
   rsync -avP --exclude="test-*.tar" \
       ../data/webdataset/ \
       vsc46423@login.hpc.ugent.be:/kyukon/scratch/gent/464/vsc46423/webdataset_shards/
   ```

2. ✅ Find your actual HPC paths by running on HPC:
   ```bash
   echo "VSC_DATA: $VSC_DATA"
   echo "VSC_SCRATCH: $VSC_SCRATCH"
   ```

   Example output:
   ```
   VSC_DATA: /data/gent/464/vsc46423
   VSC_SCRATCH: /kyukon/scratch/gent/464/vsc46423
   ```

## Configuration Steps

### 1. SudoRM-RF

Edit `src/models/sudormrf/training_config.yaml`:

**Line ~49-53** (in the `data:` section):
```yaml
  # WebDataset configuration for HPC compressed data loading
  use_webdataset: true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/webdataset_shards"  # ← Replace with your $VSC_SCRATCH path
```

**Line ~91** (in the `training:` section):
```yaml
  checkpoint_dir: "/data/gent/464/vsc46423/checkpoints/sudormrf"  # ← Replace with your $VSC_DATA path
```

### 2. TUSS

Edit `src/models/tuss/training_config.yaml`:

**Line ~62-68** (in the `data:` section):
```yaml
  # WebDataset configuration for HPC compressed data loading
  use_webdataset: true
  webdataset_path: "/kyukon/scratch/gent/464/vsc46423/webdataset_shards"  # ← Replace with your $VSC_SCRATCH path
```

**Line ~145** (in the `training:` section):
```yaml
  checkpoint_dir: "/data/gent/464/vsc46423/checkpoints/tuss"  # ← Replace with your $VSC_DATA path
```

**Line ~142** (in the `training:` section):
```yaml
  device: "cuda"  # Change from "cuda:1" to "cuda" for HPC
```

### 3. CLAPSep

Edit `scripts/hpc/train_clapsep.pbs`:

**Around line 49-56** (add the WebDataset flags):
```bash
python -u src/models/clapsep/train_coi.py \
    --df-path data/aircraft_data.csv \
    --clap-checkpoint src/models/clapsep/checkpoint/CLAPSep/model/music_audioset_epoch_15_esc_90.14.pt \
    --checkpoint-dir $VSC_DATA/checkpoints/clapsep \
    --use-webdataset \
    --webdataset-path $VSC_SCRATCH/webdataset_shards \
    --device cuda \
    --no-freeze-encoder \
    --use-lora \
    --precision bf16-mixed
```

## Verification

After making these changes, verify your setup:

1. Check that WebDataset tar files are in the right location:
   ```bash
   ls -lh $VSC_SCRATCH/webdataset_shards/
   ```
   
   You should see files like:
   ```
   train-000000.tar
   train-000001.tar
   ...
   val-000000.tar
   val-000001.tar
   ```

2. Check checkpoint directories exist:
   ```bash
   ls -ld $VSC_DATA/checkpoints/{sudormrf,tuss,clapsep}
   ```

3. Test import WebDataset utilities:
   ```bash
   module load Python/3.11.3-GCCcore-12.3.0
   source $VSC_DATA/venv/bin/activate
   export PYTHONPATH="$VSC_DATA/anthropogenic_sound_identification:$PYTHONPATH"
   python -c "from src.common.webdataset_utils import COIWebDatasetWrapper; print('WebDataset import OK')"
   ```

## Troubleshooting

### Error: "webdataset_path must be set when use_webdataset=True"
- Make sure you set `webdataset_path` in the config file with your actual HPC path
- Check that the path exists: `ls $VSC_SCRATCH/webdataset_shards/`

### Error: "No tar files found at <path>"
- Verify you transferred the WebDataset tar files correctly
- Check the path matches what you used in rsync

### Training starts but no batches load
- Check that you transferred both train AND val tar files (not just train)
- Verify tar files are not corrupted: `tar -tzf $VSC_SCRATCH/webdataset_shards/train-000000.tar | head`

## Disabling WebDataset (Fallback to File Mode)

If you need to fall back to loading individual audio files:

1. Set `use_webdataset: false` in the config YAML
2. Make sure `df_path` points to a valid metadata CSV
3. Ensure audio files are accessible at the paths listed in the CSV

Note: File mode is **much slower** on HPC because it requires reading thousands of individual files during training.
