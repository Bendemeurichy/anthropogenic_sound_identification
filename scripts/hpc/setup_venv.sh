#!/bin/bash
# =============================================================================
# HPC Virtual Environment Setup Script
# =============================================================================
# Run this INTERACTIVELY on donphan (debug cluster) for fast interactive access:
#
#   module swap cluster/donphan
#   qsub -I
#   cd $VSC_DATA/anthropogenic_sound_identification
#   bash scripts/hpc/setup_venv.sh
#
# Note: donphan uses RHEL 9, same as accelgor, so the venv will work on both.
#
# =============================================================================

set -e

echo "=============================================="
echo "Setting up Python virtual environment for HPC"
echo "=============================================="
echo ""

# Check we're on a compute node (not login node)
if [[ -z "$PBS_JOBID" ]]; then
    echo "WARNING: You don't appear to be in a PBS job."
    echo "It's recommended to run this in an interactive job on donphan:"
    echo "  module swap cluster/donphan"
    echo "  qsub -I"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Load Python module (provides base Python + pip)
echo "Loading Python module..."
module load Python/3.11.3-GCCcore-12.3.0

# Create venv directory
VENV_DIR="$VSC_DATA/venv"
echo "Virtual environment directory: $VENV_DIR"
echo ""

if [ -d "$VENV_DIR" ]; then
    echo "Warning: $VENV_DIR already exists."
    read -p "Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing venv."
        echo "To activate: source $VENV_DIR/bin/activate"
        exit 0
    fi
fi

echo "Creating virtual environment..."
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo ""
echo "Upgrading pip, wheel, setuptools..."
pip install --no-cache-dir --upgrade pip wheel setuptools

echo ""
echo "Installing requirements from requirements_hpc.txt..."
echo "  Using --no-cache-dir to avoid filling up home directory"
echo "  This may take 15-25 minutes for large packages like PyTorch..."
pip install --no-cache-dir -r scripts/hpc/requirements_hpc.txt

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Virtual environment created at: $VENV_DIR"
echo ""
echo "To activate in future sessions:"
echo "  module load Python/3.11.3-GCCcore-12.3.0"
echo "  source \$VSC_DATA/venv/bin/activate"
echo "  export PYTHONPATH=\"\$VSC_DATA/anthropogenic_sound_identification:\$PYTHONPATH\""
echo ""
echo "Creating checkpoint directories..."
mkdir -p "$VSC_DATA/checkpoints/sudormrf"
mkdir -p "$VSC_DATA/checkpoints/tuss"
mkdir -p "$VSC_DATA/checkpoints/clapsep"
echo "  Created: \$VSC_DATA/checkpoints/{sudormrf,tuss,clapsep}"
echo ""
echo "Next steps:"
echo "  1. Transfer WebDataset shards to \$VSC_SCRATCH/webdataset_shards/"
echo "  2. Edit training configs to set webdataset paths"
echo "  3. Submit jobs with: qsub scripts/hpc/train_*.pbs"
echo ""
echo "Storage info:"
echo "  - Venv location: \$VSC_DATA/venv (~5-6 GB, minimal install)"
echo "  - Checkpoints: \$VSC_DATA/checkpoints/ (will grow during training)"
echo "  - WebDataset: \$VSC_SCRATCH/webdataset_shards/ (~18 GB)"
echo ""
echo "Package savings vs full install:"
echo "  - Excluded TensorFlow (1.8 GB) - only for classifier on separate server"
echo "  - Excluded Jupyter/visualization/dev tools (~500 MB)"
echo "  - Total venv size reduced from ~11 GB to ~5-6 GB"
echo ""
