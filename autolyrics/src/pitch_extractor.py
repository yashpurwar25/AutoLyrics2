"""
AutoLyrics — Pitch Extractor
Extracts pYIN pitch (F0) track and projects to a fixed-length embedding.
"""
import numpy as np
import librosa
import torch
import torch.nn as nn

SR         = 16_000
HOP_LENGTH = 512
FMIN       = librosa.note_to_hz("C2")   # ~65 Hz — lowest singing note
FMAX       = librosa.note_to_hz("C7")   # ~2093 Hz
N_BINS     = 128                          # output pitch embedding size
N_FRAMES   = 3000                         # Whisper encoder sequence length


def extract_pitch_track(
    audio: np.ndarray, sr: int = SR
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        f0      : (T,) float array of F0 in Hz (0 where unvoiced)
        voiced  : (T,) bool voicing mask
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=FMIN, fmax=FMAX,
        sr=sr, hop_length=HOP_LENGTH,
        fill_na=0.0,
    )
    voiced = voiced_flag.astype(bool)
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32), voiced.astype(np.float32)


def pitch_to_embedding(f0: np.ndarray, target_len: int = N_FRAMES) -> np.ndarray:
    """
    Normalize F0 into [0,1], interpolate/pad to target_len,
    then expand to (target_len, N_BINS) via a Gaussian bump encoding.
    This lets the model learn smooth pitch-bin associations.
    """
    eps = 1e-8
    # Log-scale (singing F0 perception is roughly log)
    log_f0 = np.where(f0 > eps, np.log(f0 + eps), 0.0)
    lo, hi = np.log(FMIN + eps), np.log(FMAX + eps)
    norm   = (log_f0 - lo) / (hi - lo + eps)      # → [0, 1]

    # Resize to Whisper's time dimension
    if len(norm) != target_len:
        indices = np.linspace(0, len(norm) - 1, target_len)
        norm    = np.interp(indices, np.arange(len(norm)), norm)

    # Gaussian bump: each frame → a soft one-hot over N_BINS pitch bins
    bins      = np.linspace(0, 1, N_BINS)               # (N_BINS,)
    sigma     = 1.0 / N_BINS                            # bin width
    emb       = np.exp(
        -(norm[:, None] - bins[None, :]) ** 2 / (2 * sigma ** 2)
    ).astype(np.float32)                                 # (target_len, N_BINS)
    return emb


# ── Learnable projection layer ────────────────────────────────────────
class PitchProjector(nn.Module):
    """
    Projects the (T, N_BINS) pitch embedding to Whisper's hidden size.
    Inserted between pitch extractor output and encoder cross-attention.
    """
    def __init__(self, n_bins: int = N_BINS, d_model: int = 384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_bins, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, pitch_emb: torch.Tensor) -> torch.Tensor:
        "pitch_emb: (B, T, N_BINS) → (B, T, d_model)"
        return self.proj(pitch_emb)