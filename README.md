# 🎤 AutoLyrics — Singing Voice Transcription

Fine-tuning OpenAI Whisper with LoRA for music lyrics transcription.

## Results
| Model | WER | Relative Improvement |
|-------|-----|----------------------|
| Baseline (zero-shot) | 5.36% | — |
| LoRA Fine-tuned | 0.22% | **95.8% ✅** |

> Target was >15% relative WER reduction — **EXCEEDED**

## Tech Stack
Python · PyTorch · HuggingFace Transformers · PEFT/LoRA · Gradio · Librosa

## Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)

1. Open the notebook in Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
