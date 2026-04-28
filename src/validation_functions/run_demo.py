import sys
from pathlib import Path
import torch
import torchaudio

# Add paths
sys.path.insert(0, "/home/bendm/Thesis/project/code/src")
from models.tuss.inference import TUSSInference
from validation_functions.demo_separation import plot_combined_spectrograms_from_wavs


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Peak-normalize a waveform for saving to disk (human listening).
    
    Scales the audio so the peak amplitude is at target_peak (default 0.95)
    to ensure proper loudness for playback while leaving small headroom
    to prevent clipping artifacts in lossy formats.
    
    Args:
        waveform: Input waveform tensor
        target_peak: Target peak level (0.0 to 1.0), default 0.95
        
    Returns:
        Peak-normalized waveform at listening level
    """
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform
    return waveform * (target_peak / peak)


def main():
    ckpt_path = (
        "/home/bendm/Thesis/project/code/src/models/tuss/checkpoints/multi_coi_27_04"
    )
    wav_path = "/home/bendm/Thesis/project/data/misclassifications/239_as_is_sep_cls_['plane',_'wind',_'biophony']_conf0.456_S4A04430_20180716_113000.wav"

    out_dir = Path(
        "/home/bendm/Thesis/project/code/src/validation_functions/demo_output"
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading model...")
    inferencer = TUSSInference.from_checkpoint(
        ckpt_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        coi_prompt=["airplane", "birds"],
        bg_prompt="background",
    )

    print(f"Separating audio: {wav_path} with prompts {inferencer.prompts_list}")

    # Process using the now updated abstraction
    with torch.inference_mode():
        sources = inferencer.separate(wav_path)

    plane_audio = sources[0]
    bird_audio = sources[1]
    bg_audio = sources[2]

    print("Saving separated audio files...")
    import soundfile as sf

    mixture_wav_np, mix_sr = sf.read(wav_path)
    mixture_wav = torch.from_numpy(mixture_wav_np).T.to(torch.float32)

    if mix_sr != inferencer.sample_rate:
        resampler = torchaudio.transforms.Resample(mix_sr, inferencer.sample_rate)
        mixture_wav = resampler(mixture_wav)
        mix_sr = inferencer.sample_rate

    if mixture_wav.shape[0] > 1:
        mixture_wav = mixture_wav.mean(dim=0, keepdim=True)
    elif mixture_wav.dim() == 1:
        mixture_wav = mixture_wav.unsqueeze(0)

    # Save files with peak normalization for proper listening level
    mix_path = out_dir / "mixture.wav"
    plane_path = out_dir / "separated_plane.wav"
    bird_path = out_dir / "separated_bird.wav"
    bg_path = out_dir / "separated_background.wav"

    # Peak-normalize all audio for proper playback volume
    mixture_wav_norm = peak_normalize(mixture_wav.squeeze(0) if mixture_wav.dim() > 1 else mixture_wav)
    plane_audio_norm = peak_normalize(plane_audio)
    bird_audio_norm = peak_normalize(bird_audio)
    bg_audio_norm = peak_normalize(bg_audio)

    # Convert tensors back to numpy for soundfile
    sf.write(str(mix_path), mixture_wav_norm.unsqueeze(0).numpy().T, mix_sr)
    sf.write(str(plane_path), plane_audio_norm.unsqueeze(0).numpy().T, inferencer.sample_rate)
    sf.write(str(bird_path), bird_audio_norm.unsqueeze(0).numpy().T, inferencer.sample_rate)
    sf.write(str(bg_path), bg_audio_norm.unsqueeze(0).numpy().T, inferencer.sample_rate)

    print("Plotting spectrograms...")
    save_png_path = out_dir / "spectrogram_plane_bird_bg_separation.png"
    plot_combined_spectrograms_from_wavs(
        [mix_path, plane_path, bird_path, bg_path],
        save_png_path,
        titles=[
            "Original Mixture",
            "Separated Plane",
            "Separated Bird",
            "Separated Background",
        ],
        sr=inferencer.sample_rate,
        ref_idx=0,
        delta_indices=[1, 2, 3],
    )

    print(f"Successfully saved visualization to {save_png_path}")


if __name__ == "__main__":
    main()
