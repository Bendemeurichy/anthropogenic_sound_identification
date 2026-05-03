"""
Diagnostic script: inspect per-head mask magnitudes from the TUSS model.

For each source head, prints:
  - mean |mask|
  - max  |mask|
  - fraction of TF bins where |mask| > 0.1
  - fraction of TF bins where |mask| > 0.5

Also prints output RMS per head so we can cross-check against earlier runs.
"""

import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]  # code/src
TUSS_DIR = CODE_DIR / "models" / "tuss"
BASE_DIR = TUSS_DIR / "base"

for p in [str(BASE_DIR), str(CODE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.tuss.inference import TUSSInference, robust_load_audio

CHECKPOINT = TUSS_DIR / "checkpoints" / "multi_coi_29_04"
SAMPLE_RATE = 48000
SEGMENT_SAMPLES = 4 * SAMPLE_RATE  # 4 s

# Webdataset airplane sample saved from previous run
MIXTURE_WAV = CODE_DIR / "validation_functions" / "demo_output" / "webdataset_demo" / "mixture.wav"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

inferencer = TUSSInference.from_checkpoint(
    CHECKPOINT,
    device=device,
    coi_prompt=["airplane", "birds"],
    bg_prompt="background",
)
print(f"Prompts list: {inferencer.prompts_list}")
print(f"Target COI index: {inferencer.target_coi_index}")

# ---------------------------------------------------------------------------
# Monkey-patch TussModel.forward to capture raw masks
# ---------------------------------------------------------------------------
from nets.tuss import TussModel
from utils.audio_utils import do_stft

_captured_masks = {}

_original_tuss_forward = TussModel.forward


def _patched_forward(self, input: torch.Tensor, prompts):
    """Identical to original but stores raw mask before applying to STFT."""
    # Re-implement forward with mask capture
    batch0 = input.unsqueeze(-1)
    batch = torch.cat((batch0.real, batch0.imag), dim=-1)
    n_batch, n_frames, n_freqs = batch.shape[:3]
    n_src = len(prompts[0])

    batch = self.band_split_module.band_split(batch)

    if self.use_sos_token:
        sos_token = self.sos_token.unsqueeze(0).repeat(n_batch, 1, 1, self.num_bands)
        batch = torch.cat((sos_token, batch), dim=2)

    batch = self._concatenate_prompt(batch, prompts)
    for block in self.cross_prompt_module:
        batch = block(batch)

    prompt_vectors, batch = (
        batch[..., : n_src * self.prompt_size, :],
        batch[..., n_src * self.prompt_size :, :],
    )
    prompt_vectors = prompt_vectors.reshape(n_batch, -1, n_src, self.prompt_size, self.num_bands).transpose(1, 2)

    if self.use_sos_token:
        batch = batch[..., self.prompt_size :, :]

    batch = batch.unsqueeze(1).repeat(1, n_src, 1, 1, 1)
    batch = batch * prompt_vectors
    batch = batch.reshape(n_batch * n_src, -1, n_frames, self.num_bands)

    for block in self.cond_tse_module:
        batch = block(batch, n_src=n_src)

    batch = self.band_split_module.bandwise_decoding(batch)
    batch = batch.view([n_batch, n_src, 2, n_frames, n_freqs])
    batch = batch.to(torch.float32)

    # --- CAPTURE MASK HERE ---
    mask = torch.complex(batch[:, :, 0], batch[:, :, 1])  # (B, n_src, T, F)
    _captured_masks["mask"] = mask.detach().cpu()

    # Apply mask (original line 207)
    output = batch0.movedim(-1, -3) * mask
    return output


TussModel.forward = _patched_forward

# ---------------------------------------------------------------------------
# Load mixture waveform
# ---------------------------------------------------------------------------
wav, sr = robust_load_audio(MIXTURE_WAV)
print(f"\nMixture: {MIXTURE_WAV.name}, sr={sr}, shape={wav.shape}")

# Mono, resample to 48 kHz if needed
wav = wav.mean(0, keepdim=True)  # (1, T)
if sr != SAMPLE_RATE:
    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    print(f"  Resampled to {SAMPLE_RATE} Hz, new shape={wav.shape}")

# Take first full 4-second segment
seg = wav[0, :SEGMENT_SAMPLES]
if len(seg) < SEGMENT_SAMPLES:
    seg = torch.nn.functional.pad(seg, (0, SEGMENT_SAMPLES - len(seg)))

seg_tensor = seg.unsqueeze(0).to(device)  # (1, T)

# ---------------------------------------------------------------------------
# Run forward pass
# ---------------------------------------------------------------------------
model = inferencer.model
model.eval()

with torch.no_grad():
    prompts_batch = [inferencer.prompts_list]  # List[List[str]]
    # Call SeparationModel.forward — it handles STFT internally with correct params
    # The patched TussModel.forward will fire inside and capture the mask.
    Y_waves = model(seg_tensor, prompts_batch)  # (1, n_src, T)

# ---------------------------------------------------------------------------
# Analyse captured masks
# ---------------------------------------------------------------------------
if "mask" not in _captured_masks:
    print("ERROR: mask was not captured — patch may have failed")
    sys.exit(1)

mask = _captured_masks["mask"]  # (1, n_src, T, F) complex
mask_abs = mask.abs()           # (1, n_src, T, F)

print(f"\nMask tensor shape: {mask.shape}  (batch, n_src, time, freq)")
print(f"  n_src={mask.shape[1]}, n_time={mask.shape[2]}, n_freq={mask.shape[3]}")

prompts_list = inferencer.prompts_list
for i, prompt in enumerate(prompts_list):
    m = mask_abs[0, i]  # (T, F)
    mean_m = m.mean().item()
    max_m  = m.max().item()
    frac01 = (m > 0.1).float().mean().item()
    frac05 = (m > 0.5).float().mean().item()
    frac10 = (m > 1.0).float().mean().item()
    print(
        f"  Head {i} '{prompt}': "
        f"mean|mask|={mean_m:.4f}  max={max_m:.4f}  "
        f">0.1: {frac01*100:.1f}%  >0.5: {frac05*100:.1f}%  >1.0: {frac10*100:.1f}%"
    )

# Also print output RMS per head for cross-check
print("\nOutput RMS per head (dBFS):")
for i, prompt in enumerate(prompts_list):
    y_head = Y_waves[0, i]  # (T,)
    rms = y_head.pow(2).mean().sqrt().item()
    db  = 20 * np.log10(rms + 1e-12)
    print(f"  Head {i} '{prompt}': RMS={rms:.6f}  ({db:.1f} dBFS)")

# Restore original forward
TussModel.forward = _original_tuss_forward
print("\nDone.")
