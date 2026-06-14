"""
AutoLyrics — Preprocessing + Augmentation
1. Vocal isolation via spectral subtraction + harmonic mask
2. Noise reduction (noisereduce library)
3. Augmentations: pitch shift, time stretch, add instrument noise
"""
import random
import numpy as np
import librosa
import noisereduce as nr
from pathlib import Path

SR = 16_000


# ── 1. Vocal Suppression of Instruments ──────────────────────────────
def isolate_vocals(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Light vocal isolation using HPSS (Harmonic-Percussive Source Separation).
    Keeps harmonic component (vocals + melody) and discards percussive transients.
    For heavier isolation swap with Spleeter or Demucs in production.
    """
    D           = librosa.stft(audio, n_fft=2048, hop_length=512)
    H, P        = librosa.decompose.hpss(np.abs(D), margin=3.0)
    mask        = H / (H + P + 1e-8)          # soft harmonic mask
    D_harmonic  = D * mask
    vocal_audio = librosa.istft(D_harmonic, hop_length=512)
    return vocal_audio.astype(np.float32)


def reduce_background_noise(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    "Use first 0.5 s as noise profile (assumes silent intro)."
    noise_sample = audio[:sr // 2]
    return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample,
                            stationary=False, prop_decrease=0.75).astype(np.float32)


def normalize(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    return (audio / rms * target_rms).astype(np.float32)


def full_preprocess(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    audio = isolate_vocals(audio, sr)
    audio = reduce_background_noise(audio, sr)
    audio = normalize(audio)
    return audio


# ── 2. Data Augmentation ─────────────────────────────────────────────
def aug_pitch_shift(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    "Random pitch shift ±3 semitones."
    n_steps = random.uniform(-3, 3)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def aug_time_stretch(audio: np.ndarray) -> np.ndarray:
    "Random tempo shift 0.85× – 1.15×."
    rate = random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(audio, rate=rate)


def aug_add_instrument_noise(
    audio: np.ndarray,
    instrument_dir: str,
    snr_db: float = 10,
    sr: int = SR,
) -> np.ndarray:
    """
    Add a random instrument snippet at a given SNR.
    instrument_dir: folder of short wav clips (e.g. guitar, piano loops).
    This simulates the challenge of polyphonic music.
    """
    inst_dir = Path(instrument_dir)
    wav_files = list(inst_dir.glob("*.wav")) + list(inst_dir.glob("*.mp3"))
    if not wav_files:
        return audio
    noise_file = random.choice(wav_files)
    noise, _   = librosa.load(str(noise_file), sr=sr, mono=True, duration=len(audio)/sr)
    if len(noise) < len(audio):
        noise = np.pad(noise, (0, len(audio) - len(noise)))
    noise = noise[:len(audio)]
    signal_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms  = np.sqrt(np.mean(noise ** 2)) + 1e-9
    scale      = signal_rms / noise_rms / (10 ** (snr_db / 20))
    return (audio + scale * noise).astype(np.float32)


def augment(
    audio: np.ndarray,
    sr: int = SR,
    instrument_dir: Optional[str] = None,
) -> np.ndarray:
    "Randomly apply one or more augmentations."
    if random.random() < 0.5:
        audio = aug_pitch_shift(audio, sr)
    if random.random() < 0.4:
        audio = aug_time_stretch(audio)
    if instrument_dir and random.random() < 0.5:
        audio = aug_add_instrument_noise(audio, instrument_dir, sr=sr)
    return audio