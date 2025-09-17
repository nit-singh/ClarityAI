import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".m4v"}


def discover_videos(input_dir: Path) -> List[Path]:
	input_dir = Path(input_dir)
	videos: List[Path] = []
	for path in input_dir.rglob("*"):
		if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
			videos.append(path)
	return sorted(videos)


def ensure_ffmpeg_available() -> None:
	if shutil.which("ffmpeg") is None:
		raise RuntimeError(
			"ffmpeg not found on PATH. Please install ffmpeg and ensure it's accessible."
		)


def extract_audio_to_wav(video_path: Path, output_dir: Path, sample_rate: int = 16000) -> Path:
	"""Extract mono WAV audio at the given sample rate using ffmpeg."""
	ensure_ffmpeg_available()
	video_path = Path(video_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / (video_path.stem + ".wav")

	cmd = [
		"ffmpeg",
		"-y",
		"-i",
		str(video_path),
		"-ac",
		"1",
		"-ar",
		str(sample_rate),
		"-vn",
		str(output_path),
	]
	subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	return output_path
