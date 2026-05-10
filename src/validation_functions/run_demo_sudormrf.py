import sys
from pathlib import Path

import torch
import soundfile as sf

sys.path.insert(0, "/home/bendm/Thesis/project/code/src")
from models.sudormrf.inference import SeparationInference, COI_HEAD_INDEX, BACKGROUND_HEAD_INDEX
from validation_functions.demo_separation import plot_combined_spectrograms_from_wavs


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform
    return waveform * (target_peak / peak)


def main():
    ckpt_path = (
        "/home/bendm/Thesis/project/code/src/models/sudormrf/checkpoints/sudormrf_planes_10_5/best_model.pt"
    )
    wav_path = "/home/bendm/Thesis/project/data/misclassifications/239_as_is_sep_cls_['plane',_'wind',_'biophony']_conf0.456_S4A04430_20180716_113000.wav"

    out_dir = Path(
        "/home/bendm/Thesis/project/code/src/validation_functions/demo_output_sudormrf"
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading model...")
    inferencer = SeparationInference.from_checkpoint(
        ckpt_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Separating audio: {wav_path}")

    with torch.inference_mode():
        sources = inferencer.separate(wav_path)

    plane_audio = sources[COI_HEAD_INDEX]   # (T,)
    bg_audio = sources[BACKGROUND_HEAD_INDEX]  # (T,)

    print("Saving separated audio files...")
    mixture_wav_np, mix_sr = sf.read(wav_path)
    mixture_wav = torch.from_numpy(mixture_wav_np).to(torch.float32)
    if mixture_wav.dim() == 2:
        mixture_wav = mixture_wav.mean(dim=1)

    import torchaudio
    if mix_sr != inferencer.sample_rate:
        resampler = torchaudio.transforms.Resample(mix_sr, inferencer.sample_rate)
        mixture_wav = resampler(mixture_wav.unsqueeze(0)).squeeze(0)
        mix_sr = inferencer.sample_rate

    mix_path = out_dir / "mixture.wav"
    plane_path = out_dir / "separated_plane.wav"
    bg_path = out_dir / "separated_background.wav"

    sf.write(str(mix_path), peak_normalize(mixture_wav).numpy(), mix_sr)
    sf.write(str(plane_path), peak_normalize(plane_audio).numpy(), inferencer.sample_rate)
    sf.write(str(bg_path), peak_normalize(bg_audio).numpy(), inferencer.sample_rate)

    print("Plotting spectrograms...")
    save_png_path = out_dir / "spectrogram_plane_bg_separation.png"
    plot_combined_spectrograms_from_wavs(
        [mix_path, plane_path, bg_path],
        save_png_path,
        titles=["Original Mixture", "Separated Plane", "Separated Background"],
        sr=inferencer.sample_rate,
        ref_idx=0,
        delta_indices=[1, 2],
    )

    print(f"Successfully saved visualization to {save_png_path}")


if __name__ == "__main__":
    main()
