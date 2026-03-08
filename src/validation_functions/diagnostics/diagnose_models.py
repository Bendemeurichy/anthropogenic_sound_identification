import numpy as np
import torch
import torchaudio
import sys
import importlib
from pathlib import Path
from confusion_matrix import ValidationPipeline


try:
    import src.models.base.sudo_rm_rf
except Exception:
    try:
        real_mod = importlib.import_module("models.sudormrf.base.sudo_rm_rf")
        sys.modules["src.models.base.sudo_rm_rf"] = real_mod
        sys.modules["sudo_rm_rf"] = real_mod
    except Exception:
        # best-effort only; if mapping fails, fallback to normal error
        pass


def diagnostic_separation(
    checkpoint: str,
    mixture_path: str,
    gt_source_path: str,
    out_dir: str = "./separation_diagnostic",
):
    """Run diagnostics on a single mixture + ground-truth source.

    - Loads the checkpoint
    - Runs the model, extracts mask activations and separated sources
    - Computes SI-SNR between each separated source and ground-truth
    - Saves mask arrays and prints summary statistics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ValidationPipeline()
    pipeline.load_models(sep_checkpoint=checkpoint)

    # Load mixture and ground truth
    mix_wav, sr = torchaudio.load(mixture_path)
    gt_wav, sr2 = torchaudio.load(gt_source_path)
    if sr != pipeline.sample_rate:
        mix_wav = torchaudio.transforms.Resample(sr, pipeline.sample_rate)(mix_wav)
    if sr2 != pipeline.sample_rate:
        gt_wav = torchaudio.transforms.Resample(sr2, pipeline.sample_rate)(gt_wav)

    if mix_wav.shape[0] > 1:
        mix_wav = mix_wav.mean(dim=0)
    else:
        mix_wav = mix_wav.squeeze(0)

    if gt_wav.shape[0] > 1:
        gt_wav = gt_wav.mean(dim=0)
    else:
        gt_wav = gt_wav.squeeze(0)

    # Trim to same length (min)
    L = min(mix_wav.shape[0], gt_wav.shape[0])
    mix_wav = mix_wav[:L]
    gt_wav = gt_wav[:L]

    model = pipeline.separator.model

    device = pipeline.device
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        # Normalize like inference for comparable activations
        mix_mean = mix_wav.mean()
        mix_std = mix_wav.std() + 1e-8
        inp = ((mix_wav - mix_mean) / mix_std).unsqueeze(0).unsqueeze(0).to(device)

        # forward until masks
        padded = model.pad_to_appropriate_length(inp)
        enc = model.encoder(padded)
        s = enc.clone()
        x = model.ln(enc)
        x = model.bottleneck(x)
        x = model.sm(x)
        mask_pre = model.mask_net(x)
        B = mask_pre.shape[0]
        n_src = model.num_sources
        mask_pre_view = mask_pre.view(B, n_src, model.enc_num_basis, -1)

        # Model nonlinearity (typically ReLU)
        mask_post = model.mask_nl_class(mask_pre_view)

        # Analysis-only normalizations for interpretability
        mask_softmax = torch.softmax(mask_pre_view, dim=1)
        mask_sum = mask_post.sum(dim=1, keepdim=True) + 1e-8
        mask_norm = mask_post / mask_sum

        # Get separated outputs via pipeline helper
        separated = pipeline._separate(mix_wav.to(device))
        separated = separated.detach().cpu()

        # Compute STFT-domain masks from separated waveforms so plots are
        # interpretable in time-frequency space (not encoded basis space).
        # Parameters chosen for good TF resolution; adjust if needed.
        n_fft = 512
        hop_length = 128
        win_length = 512
        eps = 1e-8

        mix_spec = torch.stft(
            mix_wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True,
        )
        mix_mag = torch.abs(mix_spec)  # (freq, time)

        # separated: (n_sources, T)
        spec_mags = []
        for i in range(separated.shape[0]):
            s = separated[i]
            S = torch.stft(
                s,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                return_complex=True,
            )
            spec_mags.append(torch.abs(S))

        # Stack -> (n_sources, freq, time)
        spec_mags = torch.stack(spec_mags, dim=0)
        denom = spec_mags.sum(dim=0, keepdim=False) + eps

        for si in range(spec_mags.shape[0]):
            mask_spec = (spec_mags[si] / denom).cpu().numpy()
            np.save(out_dir / f"mask_src{si}.npy", mask_spec)
            np.save(out_dir / f"mask_src{si}_spec_mag.npy", spec_mags[si].cpu().numpy())

        # Sum across sources (spectrogram domain)
        np.save(out_dir / "masks_sum.npy", denom.cpu().numpy())

    # SI-SNR helper
    def si_snr(est, ref, eps=1e-8):
        est = est - est.mean()
        ref = ref - ref.mean()
        ref_energy = (ref**2).sum()
        if ref_energy < eps:
            return float("-inf")
        proj = (est * ref).sum() * ref / (ref_energy + eps)
        noise = est - proj
        ratio = (proj**2).sum() / ((noise**2).sum() + eps)
        return 10.0 * np.log10(ratio + eps)

    # Evaluate SI-SNR per separated source
    si_values = []
    for i in range(separated.shape[0]):
        est = separated[i][:L].numpy()
        val = si_snr(est, gt_wav[:L].numpy())
        si_values.append(val)
        # save estimated source
        torchaudio.save(
            str(out_dir / f"diag_est_src{i}.wav"),
            separated[i : i + 1, :],
            pipeline.sample_rate,
        )

    # Check separated background vs original noise (mixture - ground-truth)
    noise_si_snr = None
    try:
        orig_noise = mix_wav[:L] - gt_wav[:L]
        # choose background index: 1 if two-source model, otherwise last
        bg_idx = 1 if separated.shape[0] > 1 else separated.shape[0] - 1
        est_bg = separated[bg_idx][:L].numpy()
        noise_si_snr = si_snr(est_bg, orig_noise.numpy())
        print(f"SI-SNR (separated background vs original noise): {noise_si_snr:.3f} dB")
    except Exception:
        pass

    print("SI-SNR per estimated source:")
    for i, v in enumerate(si_values):
        print(f"  src{i}: {v:.3f} dB")

    # Save ground truth and mixture trimmed
    torchaudio.save(
        str(out_dir / "diag_mixture.wav"), mix_wav.unsqueeze(0), pipeline.sample_rate
    )
    torchaudio.save(
        str(out_dir / "diag_ground_truth.wav"),
        gt_wav.unsqueeze(0),
        pipeline.sample_rate,
    )

    return {
        "masks": [str(out_dir / f"mask_src{i}.npy") for i in range(n_src)],
        "masks_softmax": [
            str(out_dir / f"mask_src{i}_softmax.npy") for i in range(n_src)
        ],
        "masks_norm": [str(out_dir / f"mask_src{i}_norm.npy") for i in range(n_src)],
        "masks_sum": str(out_dir / "masks_sum.npy"),
        "masks_sum_softmax": str(out_dir / "masks_sum_softmax.npy"),
        "estimated_sources": [
            str(out_dir / f"diag_est_src{i}.wav") for i in range(separated.shape[0])
        ],
        "si_snr": si_values,
        "si_snr_noise": noise_si_snr,
        "mixture": str(out_dir / "diag_mixture.wav"),
        "ground_truth": str(out_dir / "diag_ground_truth.wav"),
    }


def main():
    root = Path(__file__).parent.parent
    # checkpoint = (
    #     root
    #     / "models"
    #     / "sudormrf"
    #     / "checkpoints"
    #     / "20251226_170458"
    #     / "best_model.pt"
    # )

    #base model checkpoint
    checkpoint = (
        root
        / "validation_functions"
        / "base_models"
        / "Improved_Sudormrf_U16_Bases512_WSJ02mix.pt"
    )

    mixture_path = (
        root
        / "validation_functions"
        / "separation_output_demo"
        / "mixture_20251228_124724.wav"
    )
    gt_source_path = (
        root
        / "validation_functions"
        / "separation_output_demo"
        / "source_20251228_124724.wav"
    )

    diagnostic_separation(
        checkpoint=str(checkpoint),
        mixture_path=str(mixture_path),
        gt_source_path=str(gt_source_path),
    )


if __name__ == "__main__":
    main()
    import numpy as np
    from matplotlib import pyplot as plt
    import os

    base = "./separation_diagnostic"
    files = [os.path.join(base, f"mask_src{i}.npy") for i in (0, 1)]
    masks = []
    for f in files:
        try:
            masks.append(np.load(f))
        except Exception:
            masks.append(None)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, ax in enumerate(axs):
        m = masks[i]
        if m is None:
            ax.text(0.5, 0.5, f"mask_src{i}.npy not found", ha="center", va="center")
            ax.axis("off")
            continue
        if m.ndim == 2:
            im = ax.imshow(m, cmap="gray", aspect="auto", origin="lower")
            fig.colorbar(im, ax=ax)
        else:
            ax.plot(m.flatten())
        ax.set_title(f"Mask src{i}")

    plt.tight_layout()
    plt.show()
