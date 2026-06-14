"""
AutoLyrics — Evaluation script.
Computes WER, CER on test split for baseline and fine-tuned models.
Usage: python src/evaluate.py --baseline --lora_ckpt checkpoints/lora_run1
"""
import sys; sys.path.insert(0, "src")
import argparse, json
import torch, evaluate
from datasets import DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from tqdm import tqdm
from model import PitchAwareWhisper
from pitch_extractor import extract_pitch_track, pitch_to_embedding
import numpy as np

wer_m = evaluate.load("wer")
cer_m = evaluate.load("cer")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(model, processor, dataset, use_pitch=False):
    preds, refs = [], []
    for sample in tqdm(dataset, desc="Evaluating"):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr    = sample["audio"]["sampling_rate"]
        inp   = processor(audio, sampling_rate=sr,
                          return_tensors="pt").input_features.to(DEVICE)
        pitch_emb = None
        if use_pitch:
            f0, _     = extract_pitch_track(audio, sr)
            emb       = pitch_to_embedding(f0)
            pitch_emb = torch.tensor(emb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            gen = model.generate(inp, pitch_emb=pitch_emb,
                                 language="en", task="transcribe")
        pred = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
        preds.append(pred.lower().strip())
        refs.append(sample["text"].lower().strip())
    wer = wer_m.compute(predictions=preds, references=refs)
    cer = cer_m.compute(predictions=preds, references=refs)
    return {"wer": round(wer * 100, 2), "cer": round(cer * 100, 2),
            "n_samples": len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/hf_dataset")
    parser.add_argument("--baseline_model", default="openai/whisper-small")
    parser.add_argument("--lora_ckpt", default=None)
    parser.add_argument("--output", default="results/eval_report.json")
    args = parser.parse_args()

    dd = DatasetDict.load_from_disk(args.data_dir)
    test_set = dd["test"]
    processor = WhisperProcessor.from_pretrained(args.baseline_model,
                    language="en", task="transcribe")

    results = {}

    # ── Baseline (zero-shot) ──
    print("[1/3] Zero-shot baseline...")
    baseline = PitchAwareWhisper(args.baseline_model).to(DEVICE).eval()
    results["baseline_zeroshot"] = evaluate_model(baseline, processor, test_set)

    if args.lora_ckpt:
        # ── LoRA without pitch ──
        print("[2/3] LoRA fine-tuned (no pitch)...")
        lora_model = PeftModel.from_pretrained(
            PitchAwareWhisper(args.baseline_model), args.lora_ckpt
        ).to(DEVICE).eval()
        results["lora_no_pitch"] = evaluate_model(lora_model, processor, test_set)

        # ── LoRA + pitch ──
        print("[3/3] LoRA + pitch-aware...")
        results["lora_pitch"] = evaluate_model(
            lora_model, processor, test_set, use_pitch=True
        )

    # Relative improvement
    base_wer = results["baseline_zeroshot"]["wer"]
    for k, v in results.items():
        if k != "baseline_zeroshot":
            rel = (base_wer - v["wer"]) / base_wer * 100
            v["wer_relative_improvement_%"] = round(rel, 2)

    import os; os.makedirs("results", exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\n✅ Results saved to {args.output}")
    for k, v in results.items():
        print(f"  {k}: WER={v['wer']}%  CER={v['cer']}%")


if __name__ == "__main__":
    main()