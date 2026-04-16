# HPC Training Setup Checklist

## ✅ Pre-Setup (Completed)

- [x] WebDataset compression script created
- [x] WebDataset shards created locally (~25 GB → ~18 GB without test)
- [x] HPC setup scripts created
- [x] Minimal requirements file created (saves ~5 GB)
- [x] PBS job scripts created for all 3 models

## 📋 TODO: HPC Setup Steps

### Step 1: Find Your HPC Paths
```bash
ssh login.hpc.ugent.be
echo "VSC_DATA=$VSC_DATA"
echo "VSC_SCRATCH=$VSC_SCRATCH"
# Write these down! You'll need them.
exit
```

### Step 2: Transfer Code to HPC
```bash
# Replace with your actual VSC_DATA path
rsync -avP --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    code/ login.hpc.ugent.be:<YOUR_VSC_DATA_PATH>/code/
```

### Step 3: Transfer WebDataset Shards
```bash
# Replace with your actual VSC_SCRATCH path
rsync -avP --exclude='test-*.tar' \
    data/webdataset/ \
    login.hpc.ugent.be:<YOUR_VSC_SCRATCH_PATH>/webdataset_shards/
```

**Expected transfer:** ~18 GB (train + val shards only)

### Step 4: Create Virtual Environment
```bash
ssh login.hpc.ugent.be
module swap cluster/donphan
qsub -I
cd $VSC_DATA/code
bash scripts/hpc/setup_venv.sh
# Wait 15-25 minutes
exit
```

**Expected venv size:** ~5-6 GB (minimal install)

### Step 5: Update Config Files

Edit these files with your actual paths:

**File:** `src/models/sudormrf/training_config.yaml`
```yaml
data:
  use_webdataset: true
  webdataset_path: "<YOUR_VSC_SCRATCH_PATH>/webdataset_shards"

training:
  checkpoint_dir: "<YOUR_VSC_DATA_PATH>/checkpoints/sudormrf"
```

**File:** `src/models/tuss/training_config.yaml`
```yaml
data:
  use_webdataset: true
  webdataset_path: "<YOUR_VSC_SCRATCH_PATH>/webdataset_shards"

training:
  checkpoint_dir: "<YOUR_VSC_DATA_PATH>/checkpoints/tuss"
```

**File:** `src/models/clapsep/training_config.yaml`
```yaml
data:
  use_webdataset: true
  webdataset_path: "<YOUR_VSC_SCRATCH_PATH>/webdataset_shards"

training:
  checkpoint_dir: "<YOUR_VSC_DATA_PATH>/checkpoints/clapsep"
```

### Step 6: Submit Training Jobs
```bash
ssh login.hpc.ugent.be
cd $VSC_DATA/code

# Submit jobs
qsub scripts/hpc/train_sudormrf.pbs
qsub scripts/hpc/train_tuss.pbs
qsub scripts/hpc/train_clapsep.pbs

# Check status
qstat -u $USER
```

## 📊 Storage Budget

| Location | Quota | Planned Usage |
|----------|-------|---------------|
| `$VSC_HOME` | 3 GB | < 100 MB (minimal) |
| `$VSC_DATA` | 25 GB | ~12-15 GB (code + venv + checkpoints) |
| `$VSC_SCRATCH` | 25 GB | ~18 GB (WebDataset shards) |

## 🔧 Key Optimizations Applied

1. **No pip cache** - Using `--no-cache-dir` everywhere
2. **Minimal requirements** - Excluded TensorFlow, Jupyter, viz tools
3. **No test shards** - Saved ~7 GB by excluding test data
4. **Venv in $VSC_DATA** - Not in home directory (avoids 3 GB quota)
5. **donphan for setup** - Fast interactive access vs waiting for accelgor

## ❓ Quick Commands

**Check quota:**
```bash
mmlsquota
```

**Monitor jobs:**
```bash
watch -n 30 qstat -u $USER
```

**Check logs:**
```bash
tail -f <jobname>.o<jobid>
```

**Cancel job:**
```bash
qdel <jobid>
```

## 📞 Support

- HPC issues: hpc@ugent.be
- Include `[accelgor]` in subject for GPU cluster questions
