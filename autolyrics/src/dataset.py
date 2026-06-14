"""
AutoLyrics — DatasetBuilder
Handles DALI, NUS-SVC, Jamendo, and speech fallback datasets.
Produces HuggingFace-compatible DatasetDict with 16 kHz mono audio.
"""
import json, os, random, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────
SR          = 16_000       # Whisper expects 16 kHz
MAX_SECS    = 30           # Mentor directive: clip to 30 s
MIN_SECS    = 2            # Discard clips shorter than 2 s
SEED        = 42


class LyricsDatasetBuilder:
    def __init__(
        self,
        raw_dir: str,
        out_dir: str,
        model_name: str = "openai/whisper-small",
        language: str = "en",
    ):
        self.raw_dir   = Path(raw_dir)
        self.out_dir   = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_name, language=language, task="transcribe"
        )

    # ── Low-level helpers ─────────────────────────────────────────────
    def _load_audio(self, path: str) -> Optional[np.ndarray]:
        try:
            audio, sr = librosa.load(path, sr=SR, mono=True, duration=MAX_SECS)
            if len(audio) / SR < MIN_SECS:
                return None
            return audio.astype(np.float32)
        except Exception as e:
            print(f"[WARN] Could not load {path}: {e}")
            return None

    def _make_features(self, audio: np.ndarray) -> Dict:
        mel = self.feature_extractor(
            audio, sampling_rate=SR, return_tensors="pt"
        ).input_features[0]          # shape (80, 3000)
        return mel

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer(text).input_ids

    # ── DALI parser ──────────────────────────────────────────────────
    def _parse_dali(self) -> List[Dict]:
        """
        DALI stores annotations as .gz pickle files.
        Install: pip install DALI-dataset
        """
        records = []
        dali_dir = self.raw_dir / "dali"
        if not dali_dir.exists():
            print("[INFO] DALI not found, skipping.")
            return records
        try:
            import DALI as dali_code
            dali_data = dali_code.get_the_DALI_dataset(
                str(dali_dir), sr=SR, channels=1
            )
            for annot_id, entry in tqdm(dali_data.items(), desc="DALI"):
                audio_path = dali_dir / "audio" / f"{annot_id}.wav"
                if not audio_path.exists():
                    continue
                # DALI provides note-level annotations; join to sentence level
                notes = entry.annotations["annot"]["notes"]
                text  = " ".join([n["text"] for n in notes])
                audio = self._load_audio(str(audio_path))
                if audio is None:
                    continue
                records.append({
                    "audio":  audio,
                    "text":   text.strip().lower(),
                    "source": "dali",
                })
        except ImportError:
            print("[WARN] DALI package not installed. pip install DALI-dataset")
        return records

    # ── NUS-SVC parser ───────────────────────────────────────────────
    def _parse_nus(self) -> List[Dict]:
        """
        NUS-SVC: 48 songs, audio + .txt lyric files with time alignment.
        Download: https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/
        """
        records = []
        nus_dir = self.raw_dir / "nus"
        if not nus_dir.exists():
            print("[INFO] NUS not found, skipping.")
            return records
        for lyric_file in sorted(nus_dir.rglob("*.txt")):
            audio_file = lyric_file.with_suffix(".wav")
            if not audio_file.exists():
                audio_file = lyric_file.with_suffix(".mp3")
            if not audio_file.exists():
                continue
            text  = lyric_file.read_text(encoding="utf-8").strip()
            audio = self._load_audio(str(audio_file))
            if audio is None:
                continue
            records.append({
                "audio":  audio,
                "text":   text.lower(),
                "source": "nus",
            })
        print(f"[INFO] NUS: {len(records)} records loaded.")
        return records

    # ── Jamendo parser ───────────────────────────────────────────────
    def _parse_jamendo(self) -> List[Dict]:
        """
        Jamendo Lyrics Dataset: polyphonic songs + word-level timestamps.
        Download: https://zenodo.org/record/3989267
        Mentor directive: clip each song to 30-s windows.
        """
        records = []
        jam_dir = self.raw_dir / "jamendo"
        if not jam_dir.exists():
            print("[INFO] Jamendo not found, skipping.")
            return records

        meta_file = jam_dir / "annotations" / "metadata.json"
        if not meta_file.exists():
            # Fall back to scanning individual txt files
            return self._parse_jamendo_txt(jam_dir)

        meta = json.loads(meta_file.read_text())
        for track in tqdm(meta["tracks"], desc="Jamendo"):
            audio_path = jam_dir / "audio" / f"{track['id']}.mp3"
            if not audio_path.exists():
                continue
            duration = track.get("duration", 240)
            n_clips  = max(1, math.ceil(duration / MAX_SECS))
            words    = track.get("words", [])   # [{text, start, end}, ...]

            for i in range(n_clips):
                t_start = i * MAX_SECS
                t_end   = t_start + MAX_SECS
                # Collect words falling inside window
                clip_words = [
                    w["text"] for w in words
                    if t_start <= w["start"] < t_end
                ]
                if not clip_words:
                    continue
                audio, sr = librosa.load(
                    str(audio_path), sr=SR, mono=True,
                    offset=t_start, duration=MAX_SECS
                )
                if len(audio) / SR < MIN_SECS:
                    continue
                records.append({
                    "audio":  audio.astype(np.float32),
                    "text":   " ".join(clip_words).lower(),
                    "source": "jamendo",
                })
        print(f"[INFO] Jamendo: {len(records)} 30-s clips extracted.")
        return records

    def _parse_jamendo_txt(self, jam_dir: Path) -> List[Dict]:
        "Fallback: scan .txt + matching .mp3/.wav pairs."
        records = []
        for txt in jam_dir.rglob("*.txt"):
            audio = txt.with_suffix(".mp3")
            if not audio.exists():
                audio = txt.with_suffix(".wav")
            if not audio.exists():
                continue
            text  = txt.read_text(encoding="utf-8").strip().lower()
            wav, _ = librosa.load(str(audio), sr=SR, mono=True, duration=MAX_SECS)
            if len(wav) / SR < MIN_SECS:
                continue
            records.append({"audio": wav.astype(np.float32),
                             "text": text, "source": "jamendo"})
        return records

    # ── Build + split ─────────────────────────────────────────────────
    def build(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetDict:
        all_records = (
            self._parse_dali()
            + self._parse_nus()
            + self._parse_jamendo()
        )
        if not all_records:
            raise RuntimeError("No data found. Check raw/ directory.")

        random.seed(SEED)
        random.shuffle(all_records)

        n       = len(all_records)
        n_val   = int(n * val_ratio)
        n_test  = int(n * test_ratio)
        splits  = {
            "train": all_records[n_val + n_test:],
            "val":   all_records[:n_val],
            "test":  all_records[n_val:n_val + n_test],
        }

        hf_splits = {}
        for split, recs in splits.items():
            # Save manifest
            manifest_path = self.out_dir.parent / "splits" / f"{split}.json"
            manifest_path.parent.mkdir(exist_ok=True)
            manifest = [{"text": r["text"], "source": r["source"]} for r in recs]
            manifest_path.write_text(json.dumps(manifest, indent=2))
            hf_splits[split] = Dataset.from_list([
                {"audio": {"array": r["audio"], "sampling_rate": SR},
                 "text":  r["text"], "source": r["source"]}
                for r in recs
            ])

        dd = DatasetDict(hf_splits)
        dd.save_to_disk(str(self.out_dir / "hf_dataset"))
        print(f"[INFO] Dataset saved: {n} total | train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")
        return dd


# ── DataCollator for Whisper ──────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch