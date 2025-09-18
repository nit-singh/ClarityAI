import os
from pathlib import Path
from typing import Optional, List

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .video import discover_videos, extract_audio_to_wav
from .transcribe import transcribe_audio
from .summarize import summarize_transcript
from .report import build_docx_report
from .tex import build_latex_report, compile_pdf

console = Console()


def _extract_bullets(summary_text: str, max_items: int = 15) -> List[str]:
	bullets: List[str] = []
	for line in summary_text.splitlines():
		strip = line.strip()
		if strip.startswith("- ") or strip.startswith("* "):
			bullets.append(strip[2:].strip())
			if len(bullets) >= max_items:
				break
	return bullets


def process_directory(
	input_dir: Path,
	output_dir: Path,
	model_size: str = "base",
	language: Optional[str] = None,
	gemini_api_key: Optional[str] = None,
	gemini_model: Optional[str] = None,
	write_tex: bool = True,
	compile_pdf_flag: bool = False,
	web_single: bool = False,
	bullets_json_path: Optional[Path] = None,
) -> None:
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	audio_dir = output_dir / "audio"
	transcripts_dir = output_dir / "transcripts"
	reports_dir = output_dir / "reports"
	latex_dir = output_dir / "latex"
	pdf_dir = output_dir / "pdf"
	for d in [audio_dir, transcripts_dir, reports_dir, latex_dir, pdf_dir]:
		d.mkdir(parents=True, exist_ok=True)

	videos = discover_videos(input_dir)
	if not videos:
		console.print(f"[yellow]No videos found in {input_dir}.[/yellow]")
		return

	table = Table(title="Processing Queue")
	table.add_column("#")
	table.add_column("Video")
	for idx, v in enumerate(videos, 1):
		table.add_row(str(idx), v.name)
	console.print(table)

	last_docx: Optional[Path] = None
	last_tex: Optional[Path] = None
	last_summary_text: Optional[str] = None

	for video_path in tqdm(videos, desc="Lectures", unit="video"):
		try:
			# 1) Audio extraction
			audio_path = extract_audio_to_wav(video_path, audio_dir)

			# 2) Transcription
			tr = transcribe_audio(audio_path, model_size=model_size, language=language)
			transcript_txt = transcripts_dir / f"{video_path.stem}.txt"
			transcript_txt.write_text(tr.text, encoding="utf-8")

			# 3) Summarization (Local LoRA if available; otherwise Gemini)
			summary_text = summarize_transcript(
				tr.text,
				api_key=gemini_api_key,
				model_name=gemini_model,
				title_hint=video_path.stem,
			)
			last_summary_text = summary_text

			# 4) DOCX report
			report_path = reports_dir / f"{video_path.stem}_summary.docx"
			build_docx_report(
				summary_text=summary_text,
				output_path=report_path,
				title="Lecture Summary Report",
				source_file=video_path,
			)
			console.print(f"[green]Saved DOCX:[/green] {report_path}")
			last_docx = report_path

			# 5) LaTeX report (optional)
			if write_tex:
				tex_path = latex_dir / f"{video_path.stem}_summary.tex"
				build_latex_report(
					summary_text=summary_text,
					output_path=tex_path,
					title="Lecture Summary Report",
					source_file=video_path,
				)
				console.print(f"[green]Saved LaTeX:[/green] {tex_path}")
				last_tex = tex_path
				if compile_pdf_flag:
					try:
						pdf_out = compile_pdf(tex_path)
						if pdf_out.parent != pdf_dir:
							target_pdf = pdf_dir / pdf_out.name
							pdf_out.replace(target_pdf)
							pdf_out = target_pdf
						console.print(f"[green]Saved PDF:[/green] {pdf_out}")
					except Exception as ce:
						console.print(f"[yellow]PDF compile skipped/failed:[/yellow] {ce}")

		except Exception as e:
			console.print(f"[red]Failed processing {video_path.name}:[/red] {e}")

	# Web integration: emit standard filenames and bullets JSON for the last processed video
	if web_single and last_summary_text is not None:
		std_docx = output_dir / "summary.docx"
		std_tex = output_dir / "summary.tex"
		if last_docx is not None and last_docx.exists():
			std_docx.write_bytes(last_docx.read_bytes())
		if last_tex is not None and last_tex.exists():
			std_tex.write_bytes(last_tex.read_bytes())
		bullets = _extract_bullets(last_summary_text, max_items=15)
		json_path = bullets_json_path if bullets_json_path else (output_dir / "summary.json")
		import json as _json
		json_path.write_text(_json.dumps({"bullets": bullets}, ensure_ascii=False, indent=2), encoding="utf-8")
		console.print(f"[green]Web outputs ready:[/green] {std_docx}, {std_tex}, {json_path}")


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description="Process lecture videos into summarized DOCX and LaTeX/PDF reports.")
	parser.add_argument("--input-dir", required=True, help="Directory containing lecture videos")
	parser.add_argument("--output-dir", default="outputs", help="Directory to store outputs")
	parser.add_argument("--model-size", default="base", help="faster-whisper model size (e.g., tiny, base, small, medium, large-v3)")
	parser.add_argument("--language", default=None, help="Optional ISO language code to guide transcription")
	parser.add_argument("--gemini-model", default=None, help="Gemini model name (default: gemini-1.5-flash)")
	parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
	parser.add_argument("--no-tex", action="store_true", help="Disable writing LaTeX .tex output")
	parser.add_argument("--pdf", action="store_true", help="Compile .tex to PDF via pdflatex if available")
	parser.add_argument("--web-single", action="store_true", help="Emit standard web outputs: outputs/summary.docx, summary.tex, summary.json")
	parser.add_argument("--bullets-json-path", default=None, help="Optional path for bullets JSON output")

	args = parser.parse_args()
	process_directory(
		input_dir=Path(args.input_dir),
		output_dir=Path(args.output_dir),
		model_size=args.model_size,
		language=args.language,
		gemini_api_key=args.api_key,
		gemini_model=args.gemini_model,
		write_tex=not args.no_tex,
		compile_pdf_flag=args.pdf,
		web_single=args.web_single,
		bullets_json_path=Path(args.bullets_json_path) if args.bullets_json_path else None,
	)


if __name__ == "__main__":
	main()
