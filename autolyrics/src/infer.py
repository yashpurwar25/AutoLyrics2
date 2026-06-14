"""
AutoLyrics — Inference pipeline.
Usage:
  from infer import AutoLyricsInferencer
  asr = AutoLyricsInferencer("checkpoints/lora_run1")
  text = asr.transcribe("my_song.mp3")
"""
import sys; sys.path.insert(0, "src")
import torch, librosa, numpy as np
from peft import PeftModel
from transformers import WhisperProcessor
from model import PitchAwareWhisper
from pitch_extractor import extract_pitch_track, pitch_to_embedding
from preprocess import full_preprocess

SR     = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AutoLyricsInferencer:
    def __init__(
        self,
        checkpoint_dir: str,
        base_model: str = "openai/whisper-small",
        language: str = "en",
    ):
        self.processor = WhisperProcessor.from_pretrained(
            checkpoint_dir, language=language, task="transcribe"
        )
        base  = PitchAwareWhisper(base_model)
        self.model = PeftModel.from_pretrained(base, checkpoint_dir)
        self.model.eval().to(DEVICE)

    def transcribe(
        self,
        audio_path: str,
        use_pitch: bool = True,
        preprocess: bool = True,
        chunk_secs: int = 30,
    ) -> str:
        """
        Transcribe a full song by chunking into 30-second windows.
        Returns a single joined transcript string.
        """
        audio, _ = librosa.load(audio_path, sr=SR, mono=True)
        if preprocess:
            audio = full_preprocess(audio, SR)

        chunk_size = SR * chunk_secs
        chunks     = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        transcript_parts = []

        with torch.no_grad():
            for chunk in chunks:
                if len(chunk) < SR:   # skip sub-1s tail
                    continue
                inp = self.processor(
                    chunk, sampling_rate=SR, return_tensors="pt"
                ).input_features.to(DEVICE)

                pitch_emb = None
                if use_pitch:
                    f0, _     = extract_pitch_track(chunk, SR)
                    emb       = pitch_to_embedding(f0)
                    pitch_emb = torch.tensor(emb).unsqueeze(0).to(DEVICE)

                gen  = self.model.generate(
                    inp, pitch_emb=pitch_emb,
                    language="en", task="transcribe",
                    num_beams=5, temperature=0.0,
                )
                text = self.processor.tokenizer.decode(
                    gen[0], skip_special_tokens=True
                )
                transcript_parts.append(text.strip())

        return " / ".join(transcript_parts)