from __future__ import annotations

"""
LoRA fine-tuning script for instruction-style summarization on lecture transcripts.

Usage example:
  python -m ai_lecture_summarizer.finetune_lora \
    --base_model google/gemma-2b-it \
    --train_json data/history.json \
    --output_dir outputs/lora_gemma_summarizer

Train JSON format (simple):
[
  {"instruction": "Summarize lecture ...", "input": "<transcript>", "output": "<target summary>"},
  ...
]
"""

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .video import extract_audio_to_wav, discover_videos
from .transcribe import transcribe_audio

try:
    from docx import Document as _DocxDocument  # type: ignore
except Exception:
    _DocxDocument = None  # docx parsing optional


@dataclass
class Example:
    instruction: str
    input: str
    output: str


class SummarizeDataset(Dataset):
    def __init__(self, tokenizer, data: List[Example], max_length: int = 3072):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = (
            "You are an expert note-taker for university lectures."
            " Produce a well-structured report as instructed.\n\n"
            f"Instruction: {ex.instruction}\n\n"
            f"Transcript:\n{ex.input}\n\n"
            "Answer:"
        )
        target = ex.output

        text = prompt + "\n" + target
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_examples(train_json_path: str) -> List[Example]:
    with open(train_json_path, "r", encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)
    examples: List[Example] = []
    for r in raw:
        instruction = r.get("instruction") or "Summarize the lecture transcript."
        input_text = r.get("input") or ""
        output = r.get("output") or ""
        if not input_text or not output:
            continue
        examples.append(Example(instruction=instruction, input=input_text, output=output))
    return examples


def _read_target_from_tex(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # lightly strip LaTeX commands to plain-ish text
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
    text = re.sub(r"\{\s*|\s*\}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_target_from_docx(path: Path) -> Optional[str]:
    if _DocxDocument is None:
        return None
    try:
        doc = _DocxDocument(str(path))
        paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras).strip()
    except Exception:
        return None


def _extract_index_from_name(name: str) -> Optional[str]:
    # match i<number> or o<number>(doc|tex)
    m = re.search(r"([io])(\d+)", name.lower())
    if m:
        return m.group(2)
    return None


def build_examples_from_video_and_targets(
    input_videos_dir: Path,
    target_outputs_dir: Path,
    tmp_dir: Path,
    whisper_size: str = "base",
    language: Optional[str] = None,
) -> List[Example]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = tmp_dir / "audio"
    transcripts_dir = tmp_dir / "transcripts"
    audio_dir.mkdir(exist_ok=True)
    transcripts_dir.mkdir(exist_ok=True)

    # Build map index -> target text
    index_to_target: Dict[str, str] = {}
    for p in Path(target_outputs_dir).iterdir():
        if not p.is_file():
            continue
        idx = _extract_index_from_name(p.stem)
        if not idx:
            continue
        candidate_text: Optional[str] = None
        if p.suffix.lower() == ".tex":
            candidate_text = _read_target_from_tex(p)
        elif p.suffix.lower() in {".docx", ".doc"}:
            candidate_text = _read_target_from_docx(p)
        if candidate_text:
            # prefer docx over tex if both exist; otherwise take whichever was seen first
            if idx in index_to_target and p.suffix.lower() == ".tex":
                # already have likely docx; skip overwriting with tex
                continue
            index_to_target[idx] = candidate_text

    videos = discover_videos(Path(input_videos_dir))
    examples: List[Example] = []
    for v in tqdm(videos, desc="Building training pairs (transcribe)", unit="video"):
        idx = _extract_index_from_name(v.stem)
        if not idx or idx not in index_to_target:
            continue
        # transcribe
        wav = extract_audio_to_wav(v, audio_dir)
        tr = transcribe_audio(wav, model_size=whisper_size, language=language)
        (transcripts_dir / f"{v.stem}.txt").write_text(tr.text, encoding="utf-8")
        examples.append(
            Example(
                instruction="Summarize the lecture using the requested structure.",
                input=tr.text,
                output=index_to_target[idx],
            )
        )
    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-2b-it")
    parser.add_argument("--train_json", default="data/history.json")
    parser.add_argument("--output_dir", default="outputs/lora_summarizer")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_split", type=float, default=0.2, help="Fraction for test split (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    # New dataset-from-videos options
    parser.add_argument("--from_videos", action="store_true", help="Build training data from data/input videos and data/outputs targets")
    parser.add_argument("--input_videos_dir", default="data/input", help="Directory with input videos (i1, i2, …)")
    parser.add_argument("--target_outputs_dir", default="data/outputs", help="Directory with desired outputs (o1doc.docx / o1tex.tex)")
    parser.add_argument("--tmp_dir", default="outputs/finetune_tmp", help="Temporary directory for audio/transcripts")
    parser.add_argument("--whisper_size", default="base", help="faster-whisper model size for transcription")
    parser.add_argument("--language", default=None, help="Optional language hint for transcription")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Loading training data…")
    if args.from_videos:
        train_examples = build_examples_from_video_and_targets(
            input_videos_dir=Path(args.input_videos_dir),
            target_outputs_dir=Path(args.target_outputs_dir),
            tmp_dir=Path(args.tmp_dir),
            whisper_size=args.whisper_size,
            language=args.language,
        )
        if not train_examples:
            raise RuntimeError("No matched video->target pairs found. Check naming (iN vs oNdoc/oNtex) and directories.")
    else:
        train_examples = load_examples(args.train_json)

    # Train/test split
    import random
    random.seed(args.seed)
    indices = list(range(len(train_examples)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - args.eval_split))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    train_data = [train_examples[i] for i in train_idx]
    test_data = [train_examples[i] for i in test_idx]
    print(f"Dataset size: {len(train_examples)} | Train: {len(train_data)} | Test: {len(test_data)}")
    dataset = SummarizeDataset(tokenizer, train_data, max_length=args.max_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device == "cuda" and torch.cuda.is_available()),
        dataloader_pin_memory=False,
        report_to=[],
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("Starting training…")
    trainer.train()

    print("Saving LoRA adapter…")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluation on test split
    if test_data:
        print("Evaluating on test split…")
        agg = _evaluate_adapter(
            base_model_name=args.base_model,
            adapter_dir=args.output_dir,
            test_examples=test_data,
            max_new_tokens=1024,
        )
        eval_dir = Path(args.output_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "report.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_lines = [
            f"Test samples: {agg['num_samples']}",
            f"ROUGE-1: P={agg['rouge1']['precision']:.3f} R={agg['rouge1']['recall']:.3f} F1={agg['rouge1']['f1']:.3f}",
            f"ROUGE-2: P={agg['rouge2']['precision']:.3f} R={agg['rouge2']['recall']:.3f} F1={agg['rouge2']['f1']:.3f}",
            f"ROUGE-L: P={agg['rougel']['precision']:.3f} R={agg['rougel']['recall']:.3f} F1={agg['rougel']['f1']:.3f}",
            f"BLEU: {agg['bleu']:.3f}",
            f"Accuracy@ROUGE1-F1>=0.5: {agg['accuracy_threshold']:.3f}",
            f"Structure coverage: {agg['structure_coverage']:.3f}",
        ]
        (eval_dir / "report.txt").write_text("\n".join(summary_lines), encoding="utf-8")
        print("\n".join(summary_lines))


def _evaluate_adapter(
    base_model_name: str,
    adapter_dir: str,
    test_examples: List[Example],
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rouge_score import rouge_scorer
    import sacrebleu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

    refs: List[str] = []
    hyps: List[str] = []
    rouge1_p = rouge1_r = rouge1_f = 0.0
    rouge2_p = rouge2_r = rouge2_f = 0.0
    rougel_p = rougel_r = rougel_f = 0.0
    accurate = 0
    structure_hits = 0
    sections = [
        "Executive Summary",
        "Key Takeaways",
        "Sectioned Notes",
        "Terms & Definitions",
        "Questions for Review",
    ]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for ex in tqdm(test_examples, desc="Evaluating on test set", unit="sample"):
        prompt = (
            "You are an expert note-taker for university lectures. Produce the requested structure.\n\n"
            f"Instruction: {ex.instruction}\n\nTranscript:\n{ex.input}\n\nAnswer:"
        )
        pred = generate(prompt)
        scores = scorer.score(ex.output, pred)
        r1 = scores["rouge1"]; r2 = scores["rouge2"]; rl = scores["rougeL"]
        rouge1_p += r1.precision; rouge1_r += r1.recall; rouge1_f += r1.fmeasure
        rouge2_p += r2.precision; rouge2_r += r2.recall; rouge2_f += r2.fmeasure
        rougel_p += rl.precision; rougel_r += rl.recall; rougel_f += rl.fmeasure
        if r1.fmeasure >= 0.5:
            accurate += 1
        coverage = sum(1 for s in sections if s.lower() in pred.lower()) / len(sections)
        structure_hits += coverage
        refs.append(ex.output)
        hyps.append(pred)

    n = max(1, len(test_examples))
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score / 100.0
    return {
        "num_samples": len(test_examples),
        "rouge1": {"precision": rouge1_p / n, "recall": rouge1_r / n, "f1": rouge1_f / n},
        "rouge2": {"precision": rouge2_p / n, "recall": rouge2_r / n, "f1": rouge2_f / n},
        "rougel": {"precision": rougel_p / n, "recall": rougel_r / n, "f1": rougel_f / n},
        "bleu": bleu,
        "accuracy_threshold": accurate / n,
        "structure_coverage": structure_hits / n,
    }


if __name__ == "__main__":
    main()


