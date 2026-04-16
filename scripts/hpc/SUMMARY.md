# HPC Setup Summary

## What's Been Done

All three models now have **consistent WebDataset configuration** in their YAML config files:

### 1. SudoRM-RF
- ✅ Added `use_webdataset` and `webdataset_path` fields to `src/models/sudormrf/training_config.yaml`
- ✅ PBS script uses PYTHONPATH (no editable install needed)
- ✅ Reads WebDataset settings from config file

### 2. TUSS
- ✅ Added `use_webdataset` and `webdataset_path` fields to `src/models/tuss/training_config.yaml`
- ✅ PBS script uses PYTHONPATH (no editable install needed)
- ✅ Reads WebDataset settings from config file

### 3. CLAPSep
- ✅ Added `use_webdataset` and `webdataset_path` fields to `src/models/clapsep/training_config.yaml`
- ✅ Added `--use-webdataset` and `--webdataset-path` command-line arguments to training script
- ✅ PBS script uses PYTHONPATH (no editable install needed)
- ✅ PBS script includes WebDataset flags (which override config file)

## HPC Infrastructure Files Created

```
scripts/hpc/
├── CHECKLIST.md              # Step-by-step setup checklist
├── README.md                 # Comprehensive setup guide
├── WEBDATASET_SETUP.md      # WebDataset configuration guide
├── SUMMARY.md               # This file
├── requirements_hpc.txt     # Minimal pip requirements (~5-6 GB vs 11 GB)
├── setup_venv.sh            # Virtual environment setup script
├── train_sudormrf.pbs       # SudoRM-RF training job
├── train_tuss.pbs           # TUSS training job
└── train_clapsep.pbs        # CLAPSep training job
```

## Key Design Decisions

### 1. No Editable Install
- **Problem:** Poetry's "non-package mode" prevents `pip install -e .`
- **Solution:** Use `PYTHONPATH` environment variable instead
- **Impact:** All PBS scripts set `export PYTHONPATH="$VSC_DATA/anthropogenic_sound_identification:$PYTHONPATH"`

### 2. WebDataset Configuration
- All models now have `use_webdataset` and `webdataset_path` in their config YAML files
- CLAPSep also supports command-line overrides for flexibility
- Default is `use_webdataset: false` (disabled) with empty path

### 3. Storage Optimization
- Created minimal `requirements_hpc.txt` (~5-6 GB) vs full local env (~11 GB)
- Excluded: TensorFlow, Jupyter, visualization tools, dev dependencies
- Uses `--no-cache-dir` to prevent pip cache from filling `$VSC_HOME`

### 4. Correct Directory Name
- All scripts use `anthropogenic_sound_identification` (actual repo name)
- Previously incorrectly used `code` as placeholder

## What You Need to Do on HPC

### Step 1: Get Your Paths
```bash
echo "VSC_DATA: $VSC_DATA"
echo "VSC_SCRATCH: $VSC_SCRATCH"
```

### Step 2: Transfer Files
```bash
# Transfer code (from local machine)
rsync -avP --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    /home/bendm/Thesis/project/code/ \
    vsc46423@login.hpc.ugent.be:$VSC_DATA/anthropogenic_sound_identification/

# Transfer WebDataset shards (exclude test to save ~7 GB)
rsync -avP --exclude="test-*.tar" \
    /home/bendm/Thesis/project/data/webdataset/ \
    vsc46423@login.hpc.ugent.be:$VSC_SCRATCH/webdataset_shards/
```

### Step 3: Setup Virtual Environment
```bash
# On HPC, in an interactive job on donphan
module swap cluster/donphan
qsub -I
cd $VSC_DATA/anthropogenic_sound_identification
bash scripts/hpc/setup_venv.sh
```

### Step 4: Edit Config Files
Replace placeholder paths with your actual HPC paths in:
- `src/models/sudormrf/training_config.yaml`
- `src/models/tuss/training_config.yaml`
- `src/models/clapsep/training_config.yaml`

Change:
- `use_webdataset: false` → `use_webdataset: true`
- `webdataset_path: ""` → `webdataset_path: "/actual/path/to/webdataset_shards"`
- `checkpoint_dir: "checkpoints"` → `checkpoint_dir: "/actual/path/to/checkpoints"`

### Step 5: Submit Training Jobs
```bash
qsub scripts/hpc/train_sudormrf.pbs
qsub scripts/hpc/train_tuss.pbs
qsub scripts/hpc/train_clapsep.pbs
```

## File Locations on HPC

```
$VSC_DATA/anthropogenic_sound_identification/     # Code repo
    ├── src/                                      # Source code
    ├── scripts/hpc/                              # HPC scripts
    └── ...

$VSC_DATA/venv/                                   # Virtual environment (~5-6 GB)

$VSC_DATA/checkpoints/                            # Training checkpoints
    ├── sudormrf/
    ├── tuss/
    └── clapsep/

$VSC_SCRATCH/webdataset_shards/                   # WebDataset tar files (~18 GB)
    ├── train-000000.tar
    ├── train-000001.tar
    ├── ...
    ├── val-000000.tar
    └── ...
```

## Documentation

- **CHECKLIST.md**: Quick step-by-step checklist for setup
- **README.md**: Comprehensive guide with background information
- **WEBDATASET_SETUP.md**: Detailed WebDataset configuration instructions
- **SUMMARY.md**: This overview document

## Notes

- All models use the same WebDataset format (created by `scripts/create_webdataset.py`)
- Training configs are set for 32-hour jobs on A100 GPUs (adjust if needed)
- The `df_path` field in configs is ignored when `use_webdataset: true`
- CLAPSep PBS script has WebDataset flags pre-configured (you can just submit it)
