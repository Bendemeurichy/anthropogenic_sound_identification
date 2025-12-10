"""
Training script for sudormrf model with custom serperation head and loss function.
- Uses asteroid's PITLossWrapper for automatic permutation handling
Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for aircraft (COI), 0 for background (non-COI)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from base.sudo_rm_rf.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
from src.models.sudormrf.seperation_head import (
    COISeparationHead,
    wrap_model_for_coi,
    COILoss,
)