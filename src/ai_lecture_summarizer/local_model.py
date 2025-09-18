from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class LocalModelConfig:
    base_model_name: str
    lora_adapter_dir: str
    device: str
    dtype: torch.dtype
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9


_MODEL_CACHE: dict[str, tuple[AutoModelForCausalLM, AutoTokenizer, LocalModelConfig]] = {}


def _resolve_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_local_model(
    base_model_name: Optional[str] = None,
    lora_adapter_dir: Optional[str] = None,
) -> Optional[tuple[AutoModelForCausalLM, AutoTokenizer, LocalModelConfig]]:
    base_model = base_model_name or os.getenv("LOCAL_MODEL_NAME", "google/gemma-2b-it")
    adapter_dir = lora_adapter_dir or os.getenv("LORA_ADAPTER_DIR", os.path.join("outputs", "lora_summarizer"))

    if not (adapter_dir and os.path.isdir(adapter_dir)):
        return None

    cache_key = f"{base_model}::{adapter_dir}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    device, dtype = _resolve_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    cfg = LocalModelConfig(
        base_model_name=base_model,
        lora_adapter_dir=adapter_dir,
        device=device,
        dtype=dtype,
    )
    _MODEL_CACHE[cache_key] = (model, tokenizer, cfg)
    return _MODEL_CACHE[cache_key]


def generate_summary_with_local_model(
    transcript_text: str,
    title_hint: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_chunk_chars: int = 6000,
) -> Optional[str]:
    loaded = load_local_model()
    if loaded is None:
        return None
    model, tokenizer, cfg = loaded

    def chunk_text(text: str) -> List[str]:
        if len(text) <= max_chunk_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chunk_chars, len(text))
            boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary <= start + 1000:
                boundary = end
            else:
                boundary += 1
            chunks.append(text[start:boundary].strip())
            start = boundary
        return chunks

    chunks = chunk_text(transcript_text)

    def make_prompt(prefix: str, content: str) -> str:
        sys = system_prompt or (
            "You are an expert note-taker for university lectures. Produce a clear, well-structured report."
        )
        title = f"\n\nLecture title/topic hint: {title_hint}" if title_hint else ""
        return f"{sys}{title}\n\n{prefix}\n\n{content}\n\nAnswer:"

    partials: List[str] = []
    for i, ch in enumerate(chunks, 1):
        prompt = make_prompt(f"Transcript chunk {i}/{len(chunks)}:", ch)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Heuristically strip the prompt prefix
        partial = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        partials.append(partial)

    if len(partials) == 1:
        synthesis_input = partials[0]
    else:
        synthesis_input = "\n\n".join(partials)

    synth_prompt = make_prompt(
        "Synthesize a single cohesive report from the partial summaries. Merge overlaps and keep structure.",
        synthesis_input,
    )
    synth_inputs = tokenizer(synth_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        synth_outputs = model.generate(
            **synth_inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    synth_text = tokenizer.decode(synth_outputs[0], skip_special_tokens=True)
    final = synth_text[len(synth_prompt):].strip() if synth_text.startswith(synth_prompt) else synth_text.strip()
    return final


