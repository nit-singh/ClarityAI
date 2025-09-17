from __future__ import annotations

import os
from typing import List, Optional

import google.generativeai as genai

_DEFAULT_MODEL = "gemini-1.5-flash"

SYSTEM_PROMPT = (
	"You are an expert note-taker for university lectures. Produce a clear, well-structured, and moderately detailed report with: "
	"1) Executive Summary (5-8 sentences). "
	"2) Key Takeaways (8-15 concise bullets). "
	"3) Sectioned Notes: Use headings (## Heading) followed by short bullets. "
	"4) Terms & Definitions: important terminology (3-10 items). "
	"5) Practical Examples or Applications if discussed. "
	"6) Questions for Review (4-8 items). "
	"Use an academic tone, remove filler, avoid repetition, and maintain coherence across sections."
)


def _configure(api_key: Optional[str]) -> None:
	key = api_key or os.getenv("GEMINI_API_KEY")
	if not key:
		raise ValueError("Gemini API key not provided. Pass --api-key or set GEMINI_API_KEY.")
	genai.configure(api_key=key)


def _model(model_name: Optional[str]) -> genai.GenerativeModel:
	return genai.GenerativeModel(model_name or _DEFAULT_MODEL)


def _chunk_text(text: str, max_chars: int = 12000) -> List[str]:
	if len(text) <= max_chars:
		return [text]
	chunks: List[str] = []
	start = 0
	while start < len(text):
		end = min(start + max_chars, len(text))
		boundary = text.rfind(". ", start, end)
		if boundary == -1 or boundary <= start + 2000:
			boundary = end
		else:
			boundary += 1
		chunks.append(text[start:boundary].strip())
		start = boundary
	return chunks


def summarize_transcript(
	transcript_text: str,
	api_key: Optional[str] = None,
	model_name: Optional[str] = None,
	title_hint: Optional[str] = None,
) -> str:
	"""Summarize a transcript using Gemini, handling long inputs via map-reduce."""
	_configure(api_key)
	model = _model(model_name)
	chunks = _chunk_text(transcript_text)

	partial_summaries: List[str] = []
	for idx, chunk in enumerate(chunks, 1):
		prompt = (
			f"{SYSTEM_PROMPT}\n\nTranscript chunk {idx}/{len(chunks)}:\n\n" + chunk
		)
		resp = model.generate_content(prompt)
		partial_summaries.append(resp.text.strip())

	if len(partial_summaries) == 1:
		final_input = partial_summaries[0]
	else:
		final_input = "\n\n".join(partial_summaries)

	final_prompt = (
		f"{SYSTEM_PROMPT}\n\nSynthesize a single cohesive report from the partial summaries. "
		"Merge overlapping content, remove duplicates, and keep the requested structure."
	)
	if title_hint:
		final_prompt += f"\n\nLecture title/topic hint: {title_hint}"

	resp = model.generate_content([final_prompt, final_input])
	return resp.text.strip()
