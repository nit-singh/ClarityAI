from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel


@dataclass
class TranscriptSegment:
	start: float
	end: float
	text: str


@dataclass
class TranscriptResult:
	text: str
	segments: List[TranscriptSegment]
	language: Optional[str]


def transcribe_audio(
	audio_path: Path,
	model_size: str = "base",
	device: str = "cpu",
	compute_type: str = "int8",
	language: Optional[str] = None,
) -> TranscriptResult:
	model = WhisperModel(model_size, device=device, compute_type=compute_type)
	segments, info = model.transcribe(
		str(audio_path),
		language=language,
		temperature=0.0,
	)
	collected: List[TranscriptSegment] = []
	texts: List[str] = []
	for seg in segments:
		collected.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip()))
		texts.append(seg.text.strip())
	full_text = " ".join(texts)
	return TranscriptResult(text=full_text, segments=collected, language=getattr(info, "language", None))
