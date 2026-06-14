"""
AutoLyrics — PitchAwareWhisper
Wraps openai/whisper-* and injects a PitchProjector output
into the encoder hidden states via additive fusion.
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperConfig

from pitch_extractor import PitchProjector, N_BINS


class PitchAwareWhisper(nn.Module):
    """
    PitchAwareWhisper = Whisper encoder-decoder + PitchProjector.

    At each forward pass:
      1. Whisper encoder converts mel spectrogram → encoder_hidden (B, T, d)
      2. PitchProjector converts pitch embedding → pitch_hidden (B, T, d)
      3. encoder_hidden += alpha * pitch_hidden   (learnable scalar alpha)
      4. Whisper decoder generates tokens as normal.

    Only PitchProjector + alpha are trained from scratch;
    Whisper weights are fine-tuned via LoRA.
    """
    def __init__(self, model_name: str = "openai/whisper-small"):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name)
        d_model      = self.whisper.config.d_model
        self.pitch_proj = PitchProjector(n_bins=N_BINS, d_model=d_model)
        self.alpha      = nn.Parameter(torch.tensor(0.1))  # starts small

    def forward(
        self,
        input_features: torch.Tensor,           # (B, 80, 3000)
        pitch_emb: Optional[torch.Tensor],       # (B, 3000, N_BINS) or None
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ):
        # Run Whisper encoder manually to access hidden states
        encoder_out = self.whisper.model.encoder(
            input_features=input_features,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = encoder_out.last_hidden_state   # (B, T, d)

        if pitch_emb is not None:
            pitch_h  = self.pitch_proj(pitch_emb)   # (B, T, d)
            # Align T dimensions (encoder downsamples 2×)
            T_enc    = hidden.shape[1]
            pitch_h  = pitch_h[:, ::2, :][:, :T_enc, :]  # stride-match
            hidden   = hidden + self.alpha * pitch_h

        # Pass fused hidden states to decoder
        outputs = self.whisper(
            encoder_outputs=(hidden,),
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        return outputs

    def generate(self, input_features, pitch_emb=None, **kwargs):
        encoder_out = self.whisper.model.encoder(input_features=input_features, return_dict=True)
        hidden = encoder_out.last_hidden_state
        if pitch_emb is not None:
            T_enc   = hidden.shape[1]
            pitch_h = self.pitch_proj(pitch_emb)[:, ::2, :][:, :T_enc, :]
            hidden  = hidden + self.alpha * pitch_h
        return self.whisper.generate(encoder_outputs=(hidden,), **kwargs)