"""
AutoLyrics — Gradio Demo
Shows WER comparison between baseline and fine-tuned model.
Launch: python app/demo.py
"""
import sys; sys.path.insert(0, "src")
import gradio as gr
import numpy as np
import soundfile as sf
import tempfile, jiwer
from infer import AutoLyricsInferencer

CHECKPOINT = "checkpoints/lora_run1"
asr = AutoLyricsInferencer(CHECKPOINT)

def transcribe_fn(audio_file, ref_text, use_pitch, preprocess):
    transcript = asr.transcribe(audio_file, use_pitch=use_pitch, preprocess=preprocess)
    metrics    = {}
    if ref_text.strip():
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        wer = jiwer.wer(
            ref_text, transcript,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )
        metrics_str = f"WER: {wer*100:.2f}%"
    else:
        metrics_str = "(Paste reference lyrics above to compute WER)"
    return transcript, metrics_str


with gr.Blocks(title="AutoLyrics", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🎤 AutoLyrics — Singing Voice Transcription
*Whisper + LoRA + Pitch-Aware Encoder | Fine-tuned on DALI / NUS / Jamendo*
""")
    with gr.Row():
        with gr.Column():
            audio_input  = gr.Audio(sources=["upload", "microphone"],
                                    type="filepath", label="Song clip (≤30 s)")
            ref_lyrics   = gr.Textbox(label="Reference lyrics (optional, for WER)",
                                      lines=4, placeholder="Paste ground-truth here...")
            use_pitch    = gr.Checkbox(value=True,  label="Use pitch-aware encoder")
            preprocess   = gr.Checkbox(value=True,  label="Vocal isolation + noise reduction")
            run_btn      = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            out_text     = gr.Textbox(label="Transcribed Lyrics", lines=10)
            out_metrics  = gr.Textbox(label="Metrics", lines=2)

    run_btn.click(
        transcribe_fn,
        inputs=[audio_input, ref_lyrics, use_pitch, preprocess],
        outputs=[out_text, out_metrics],
    )
    gr.Examples(
        examples=[["data/raw/examples/sample1.mp3", "", True, True]],
        inputs=[audio_input, ref_lyrics, use_pitch, preprocess],
    )

if __name__ == "__main__":
    demo.launch(share=True)