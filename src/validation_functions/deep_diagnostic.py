"""
Deep diagnostic for separation model - checks internal activations and separation quality.
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from confusion_matrix import ValidationPipeline


def deep_diagnostic(checkpoint: str, mixture_path: str, gt_source_path: str = None):
    """Run comprehensive diagnostics on model internals."""

    pipeline = ValidationPipeline()
    pipeline.load_models(sep_checkpoint=checkpoint)

    # Load mixture
    mix_wav, sr = torchaudio.load(mixture_path)
    if sr != pipeline.sample_rate:
        mix_wav = torchaudio.transforms.Resample(sr, pipeline.sample_rate)(mix_wav)
    if mix_wav.shape[0] > 1:
        mix_wav = mix_wav.mean(dim=0)
    else:
        mix_wav = mix_wav.squeeze(0)

    L = min(mix_wav.shape[0], int(pipeline.sample_rate * 5.0))
    mix_wav = mix_wav[:L]

    model = pipeline.separator.model
    device = pipeline.device
    model.eval()

    print("=" * 80)
    print("DEEP DIAGNOSTIC REPORT")
    print("=" * 80)

    with torch.inference_mode():
        # Normalize input
        mix_mean = mix_wav.mean()
        mix_std = mix_wav.std() + 1e-8
        inp = ((mix_wav - mix_mean) / mix_std).unsqueeze(0).unsqueeze(0).to(device)

        print(f"\n1. Input statistics:")
        print(f"   Shape: {inp.shape}")
        print(f"   Mean: {inp.mean():.4f}, Std: {inp.std():.4f}")
        print(f"   Min: {inp.min():.4f}, Max: {inp.max():.4f}")

        # Encoder
        padded = model.pad_to_appropriate_length(inp)
        enc = model.encoder(padded)
        print(f"\n2. Encoder output:")
        print(f"   Shape: {enc.shape}")
        print(f"   Mean: {enc.mean():.4f}, Std: {enc.std():.4f}")
        print(f"   Min: {enc.min():.4f}, Max: {enc.max():.4f}")
        print(f"   Energy (L2): {enc.pow(2).sum().sqrt():.4f}")

        # After LayerNorm
        s = enc.clone()
        x = model.ln(enc)
        print(f"\n3. After LayerNorm:")
        print(f"   Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        print(f"   Min: {x.min():.4f}, Max: {x.max():.4f}")

        # Bottleneck
        x = model.bottleneck(x)
        print(f"\n4. After Bottleneck:")
        print(f"   Shape: {x.shape}")
        print(f"   Mean: {x.mean():.4f}, Std: {x.std():.4f}")

        # Separation module
        x = model.sm(x)
        print(f"\n5. After Separation Module:")
        print(f"   Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        print(f"   Min: {x.min():.4f}, Max: {x.max():.4f}")

        # Mask network pre-nonlinearity
        mask_pre = model.mask_net(x)
        B = mask_pre.shape[0]
        n_src = model.num_sources
        mask_pre_view = mask_pre.view(B, n_src, model.enc_num_basis, -1)

        print(f"\n6. Mask pre-nonlinearity (raw from mask_net):")
        print(f"   Shape: {mask_pre_view.shape}")
        for i in range(n_src):
            m = mask_pre_view[0, i]
            print(
                f"   Source {i}: mean={m.mean():.4f}, std={m.std():.4f}, min={m.min():.4f}, max={m.max():.4f}"
            )

        # After mask nonlinearity
        mask_post = model.mask_nl_class(mask_pre_view)
        print(f"\n7. Mask post-nonlinearity (ReLU):")
        for i in range(n_src):
            m = mask_post[0, i]
            sparsity = (m == 0).float().mean()
            print(
                f"   Source {i}: mean={m.mean():.4f}, std={m.std():.4f}, sparsity={sparsity:.2%}"
            )

        # Check mask complementarity
        mask_sum = mask_post.sum(dim=1)[0]
        print(f"\n8. Mask complementarity (sum across sources):")
        print(f"   Mean: {mask_sum.mean():.4f}, Std: {mask_sum.std():.4f}")
        print(f"   Min: {mask_sum.min():.4f}, Max: {mask_sum.max():.4f}")

        # Softmax across sources
        mask_softmax = torch.softmax(mask_pre_view, dim=1)
        print(f"\n9. Softmax masks (forced complementarity):")
        for i in range(n_src):
            m = mask_softmax[0, i]
            print(f"   Source {i}: mean={m.mean():.4f}, std={m.std():.4f}")
            # Check if it's close to uniform (0.5 for 2 sources)
            expected_uniform = 1.0 / n_src
            print(
                f"   Distance from uniform ({expected_uniform:.3f}): {abs(m.mean() - expected_uniform):.4f}"
            )

        # Masked encoder
        masked = mask_post * s.unsqueeze(1)
        print(f"\n10. After masking encoder features:")
        for i in range(n_src):
            m = masked[0, i]
            print(f"   Source {i}: mean={m.mean():.4f}, std={m.std():.4f}")

        # Full forward
        estimates = model(inp)
        print(f"\n11. Final separated waveforms:")
        print(f"   Shape: {estimates.shape}")
        for i in range(estimates.shape[1]):
            wav = estimates[0, i]
            print(f"   Source {i}: mean={wav.mean():.4f}, std={wav.std():.4f}")
            print(f"   Source {i}: min={wav.min():.4f}, max={wav.max():.4f}")

        # If ground truth available, compute SI-SNR
        if gt_source_path:
            gt_wav, sr2 = torchaudio.load(gt_source_path)
            if sr2 != pipeline.sample_rate:
                gt_wav = torchaudio.transforms.Resample(sr2, pipeline.sample_rate)(
                    gt_wav
                )
            if gt_wav.shape[0] > 1:
                gt_wav = gt_wav.mean(dim=0)
            else:
                gt_wav = gt_wav.squeeze(0)
            gt_wav = gt_wav[:L]

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

            print(f"\n12. Separation quality (SI-SNR vs ground truth):")
            for i in range(estimates.shape[1]):
                est = estimates[0, i].cpu().numpy()
                val = si_snr(est, gt_wav.numpy())
                print(f"   Source {i}: {val:.3f} dB")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    # Analyze the results
    mask_mean_diff = abs(mask_softmax[0, 0].mean() - 0.5)
    if mask_mean_diff < 0.05:
        print("⚠️  CRITICAL: Softmax masks are nearly uniform (mean ≈ 0.5)")
        print("   This means the model is NOT learning to separate sources.")
        print("   The model is essentially outputting equal masks everywhere.")

    if mask_post.std() < 0.1:
        print("⚠️  WARNING: Very low mask variance")
        print("   Masks have minimal variation across time/frequency bins.")

    print("\nRECOMMENDATIONS:")
    print("1. Check training loss - should be decreasing steadily")
    print("2. Verify training data has actual source separation examples")
    print("3. Consider lower learning rate or different optimizer")
    print("4. Check if class_weight in loss is too extreme")
    print("5. Verify ground truth sources are aligned with mixtures")
    print("=" * 80)


def main():
    root = Path(__file__).parent.parent
    checkpoint = (
        root
        / "models"
        / "sudormrf"
        / "checkpoints"
        / "20251217_120142"
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

    deep_diagnostic(
        checkpoint=str(checkpoint),
        mixture_path=str(mixture_path),
        gt_source_path=str(gt_source_path),
    )


if __name__ == "__main__":
    main()
