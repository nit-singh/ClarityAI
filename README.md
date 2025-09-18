# Nitish Kumar singh
# Mechanical Engineering
# IIT Guwahati

# AI Lecture Summarizer

Process a folder of lecture videos into structured Word and LaTeX reports using open-source transcription (faster-whisper) and Gemini for summarization.

## Features
- Multi-format video input: `.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`
- Audio extraction via ffmpeg
- Speech-to-text with faster-whisper
- Gemini-powered, richer summaries
- LaTeX-like structured `.docx` report via python-docx
- Optional `.tex` LaTeX report and PDF compilation

## Requirements
- Python 3.10+
- ffmpeg installed and on PATH
  - Windows: Download from `https://www.gyan.dev/ffmpeg/builds/`, unzip, and add `bin` to PATH.
- Gemini API key
- (Optional) LaTeX distribution to compile PDF (`pdflatex` on PATH)
  - Windows: MiKTeX `https://miktex.org/download`
  - macOS: MacTeX `https://tug.org/mactex/`
  - Linux: TeX Live via your package manager

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your Gemini API key:
```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

## Usage
```powershell
python main.py --input-dir "D:\\lectures" --output-dir "D:\\project\\outputs" --model-size base --gemini-model gemini-1.5-flash --pdf
```
Flags:
- `--no-tex`: disable LaTeX `.tex` output
- `--pdf`: attempt to compile `.tex` to PDF using `pdflatex`

Outputs are saved under `outputs/`:
- `audio/` extracted WAV files
- `transcripts/` transcript `.txt`
- `reports/` Word documents
- `latex/` LaTeX `.tex` files
- `pdf/` compiled PDFs (if `--pdf`)

## Notes
- For faster transcription on GPU, install CUDA-enabled dependencies per faster-whisper docs and pick an appropriate model.
- Summaries are slightly longer and include sections for key takeaways, terminology, and review questions.
- To guide transcription language, use `--language en` (ISO code).

## Web App (optional)

Setup:
```powershell
npm install
```
Run:
```powershell
npm run start
```
Open `http://localhost:3000`, upload a video, optionally provide `Gemini API Key`, and click Summarize. Downloads appear when ready.

The server runs the Python agent as:
```powershell
python main.py --input-dir uploads --output-dir outputs --model-size base --gemini-model gemini-1.5-flash --web-single
```
If `GEMINI_API_KEY` is not set on the server, you can pass one per request via the `x-gemini-key` header from the web UI.
