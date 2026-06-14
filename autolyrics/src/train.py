"""
AutoLyrics — Training script.
Usage: python src/train.py --config configs/lora_finetune.yaml
"""
import sys, os
sys.path.insert(0, "src")
import argparse
import numpy as np
import evaluate
import torch
from omegaconf import OmegaConf
from datasets import DatasetDict
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer,
)

from lora_config import build_lora_model
from dataset import DataCollatorSpeechSeq2SeqWithPadding
from pitch_extractor import extract_pitch_track, pitch_to_embedding

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def prepare_dataset(batch, processor, tokenizer):
    audio = batch["audio"]
    arr   = audio["array"]
    sr    = audio["sampling_rate"]
    # Mel features
    batch["input_features"] = processor.feature_extractor(
        arr, sampling_rate=sr
    ).input_features[0]
    # Pitch embedding
    f0, _   = extract_pitch_track(arr, sr)
    pitch_e = pitch_to_embedding(f0)               # (3000, 128)
    batch["pitch_emb"] = pitch_e.tolist()
    # Labels
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def compute_metrics(pred, tokenizer):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str   = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4), "cer": round(cer, 4)}


def main(cfg_path: str):
    cfg   = OmegaConf.load(cfg_path)
    train_cfg = cfg.training

    processor = WhisperProcessor.from_pretrained(
        cfg.model_name, language=cfg.language, task="transcribe"
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model_name, language=cfg.language, task="transcribe"
    )
    model = build_lora_model(cfg.model_name)
    model.whisper.config.forced_decoder_ids = None
    model.whisper.config.suppress_tokens    = []

    dd = DatasetDict.load_from_disk(cfg.data_dir)
    dd = dd.map(
        lambda b: prepare_dataset(b, processor, tokenizer),
        remove_columns=["audio", "text", "source"],
        num_proc=4,
    )

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.whisper.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = cfg.output_dir,
        per_device_train_batch_size = train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps = train_cfg.gradient_accumulation_steps,
        learning_rate               = train_cfg.learning_rate,
        warmup_steps                = train_cfg.warmup_steps,
        max_steps                   = train_cfg.max_steps,
        fp16                        = train_cfg.fp16,
        evaluation_strategy         = train_cfg.evaluation_strategy,
        eval_steps                  = train_cfg.eval_steps,
        save_steps                  = train_cfg.save_steps,
        logging_steps               = train_cfg.logging_steps,
        predict_with_generate       = True,
        generation_max_length       = 225,
        load_best_model_at_end      = train_cfg.load_best_model_at_end,
        metric_for_best_model       = train_cfg.metric_for_best_model,
        greater_is_better           = train_cfg.greater_is_better,
        report_to                   = train_cfg.report_to,
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = dd["train"],
        eval_dataset  = dd["val"],
        tokenizer     = processor.feature_extractor,
        data_collator = collator,
        compute_metrics = lambda p: compute_metrics(p, tokenizer),
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_finetune.yaml")
    args = parser.parse_args()
    main(args.config)