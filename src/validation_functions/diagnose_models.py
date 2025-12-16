import numpy as np
import torch
import torchaudio
from pathlib import Path
from confusion_matrix import ValidationPipeline


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
        inp = mix_wav.unsqueeze(0).unsqueeze(0).to(device)
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
        mask_post = model.mask_nl_class(mask_pre_view)

        # Save mask stats and arrays
        for si in range(n_src):
            arr = mask_post[0, si].cpu().numpy()
            np.save(out_dir / f"mask_src{si}.npy", arr)
            print(
                f"mask_src{si}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
            )

        # Sum across sources (should indicate complementarity)
        sum_masks = mask_post.sum(dim=1)[0].cpu().numpy()
        np.save(out_dir / "masks_sum.npy", sum_masks)
        print(
            f"masks_sum: min={sum_masks.min():.4f}, max={sum_masks.max():.4f}, mean={sum_masks.mean():.4f}"
        )

        # Get separated outputs via pipeline helper
        separated = pipeline._separate(mix_wav.to(device))
        separated = separated.detach().cpu()

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
        "masks_sum": str(out_dir / "masks_sum.npy"),
        "estimated_sources": [
            str(out_dir / f"diag_est_src{i}.wav") for i in range(separated.shape[0])
        ],
        "si_snr": si_values,
        "mixture": str(out_dir / "diag_mixture.wav"),
        "ground_truth": str(out_dir / "diag_ground_truth.wav"),
    }


def main():
    root = Path(__file__).parent.parent
    checkpoint = (
        root
        / "models"
        / "sudormrf"
        / "checkpoints"
        / "20251215_234806"
        / "best_model.pt"
    )

    mixture_path = (
        root
        / "validation_functions"
        / "separation_output_demo"
        / "mixture_20251216_155938.wav"
    )
    gt_source_path = (
        root
        / "validation_functions"
        / "separation_output_demo"
        / "source_20251216_155938.wav"
    )

    diagnostic_separation(
        checkpoint=str(checkpoint),
        mixture_path=str(mixture_path),
        gt_source_path=str(gt_source_path),
    )


if __name__ == "__main__":
    main()
