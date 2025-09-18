## Nitish Kumar singh
## Mechanical Engineering
## IIT Guwahati
# AI Lecture Summarizer

Process lecture videos into structured reports. The pipeline: ffmpeg audio extraction → faster-whisper transcription → summarization via local LoRA adapter or Gemini → export to Word (.docx) and optional LaTeX/PDF. A minimal web UI is included.

## Highlights
- Multi-format video input: `.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`
- Audio extraction via ffmpeg
- Transcription with faster-whisper
- Summarization with either:
  - Local LoRA adapter on a base LLM (e.g., `google/gemma-2b-it`), or
  - Gemini API (`gemini-1.5-flash` by default)
- Exports: `.docx` with Python-docx; optional `.tex` and PDF
- Optional web interface (Node/Express + static HTML/JS)

---

## Project Structure
```text
.
├─ main.py                         # Entrypoint that delegates to CLI
├─ src/ai_lecture_summarizer/
│  ├─ cli.py                       # CLI to process directories of videos
│  ├─ summarize.py                 # Summarization orchestrator (LoRA or Gemini)
│  ├─ local_model.py               # Local LoRA loading/generation helpers
│  ├─ finetune_lora.py             # LoRA fine-tuning + evaluation
│  ├─ transcribe.py                # faster-whisper transcription
│  ├─ video.py                     # ffmpeg integration
│  ├─ report.py                    # .docx builder
│  ├─ tex.py                       # .tex builder and PDF compile
│  └─ __init__.py
├─ server/index.js                 # Minimal web server
├─ public/{index.html,main.js,styles.css}
├─ outputs/                        # Default output directory
└─ data/                           # Example input/target folders for fine-tune
```

---

## Installation
- Python 3.10+
- ffmpeg on PATH
  - Windows: download from `https://www.gyan.dev/ffmpeg/builds/`, unzip, add `bin` to PATH
- (Optional) LaTeX distribution for PDF (`pdflatex` on PATH)
  - Windows: MiKTeX (`https://miktex.org/download`)
  - macOS: MacTeX (`https://tug.org/mactex/`)
  - Linux: TeX Live (package manager)

Python deps:
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your Gemini API key if you will use Gemini mode:
```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

---

## Usage (CLI)
The CLI processes a directory of lecture videos.

```powershell
python main.py --input-dir "D:\\lectures" --output-dir "D:\\project\\outputs" --model-size base --gemini-model gemini-1.5-flash --pdf
```

Flags:
- `--model-size`: faster-whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `--language`: optional ISO code to guide transcription (e.g. `en`)
- `--gemini-model`: Gemini model name (default: `gemini-1.5-flash`)
- `--api-key`: Gemini API key (or set `GEMINI_API_KEY`)
- `--no-tex`: disable LaTeX `.tex` output
- `--pdf`: attempt to compile `.tex` to PDF using `pdflatex`
- `--web-single`: emit standard names for web UI (`outputs/summary.docx|.tex|.json`)

Outputs under `outputs/`:
- `audio/`: extracted WAV files
- `transcripts/`: transcript `.txt`
- `reports/`: Word documents
- `latex/`: LaTeX `.tex` files
- `pdf/`: compiled PDFs (if `--pdf`)
- `summary.docx`, `summary.tex`, `summary.json`: last processed file (when `--web-single`)

Performance notes:
- For GPU transcription, install CUDA-enabled faster-whisper and select a larger model.

---

## Summarization Modes
Summarization is controlled by `SUMMARIZER_MODE` env var in `src/ai_lecture_summarizer/summarize.py`.

- `SUMMARIZER_MODE=lora` — force local LoRA adapter
  - Requires a fine-tuned adapter directory (see Fine-tuning), configured via `LORA_ADAPTER_DIR`.
- `SUMMARIZER_MODE=gemini` — always use Gemini API
- unset/`auto` — prefer LoRA if available; otherwise fallback to Gemini

Environment variables for local inference (see `local_model.py`):
- `LOCAL_MODEL_NAME` (default: `google/gemma-2b-it`)
- `LORA_ADAPTER_DIR` (default: `outputs/lora_summarizer`)

Prompting: The system prompt produces a report with sections: Executive Summary, Key Takeaways, Sectioned Notes (## headings + bullets), Terms & Definitions, Practical Examples, and Questions for Review.

---

## Fine-tuning (LoRA)
All fine-tuning code lives in `src/ai_lecture_summarizer/finetune_lora.py`. It trains a LoRA adapter on a base chat/instruction model (default `google/gemma-2b-it`) for instruction-style lecture summarization.

### Data formats
You can supply training data in two ways:

1) JSON file (simple instruction tuning):
```json
[
  { "instruction": "Summarize lecture ...", "input": "<transcript>", "output": "<target summary>" }
]
```

2) Auto-build from videos + target outputs:
- Place input videos under `data/input/` as `i1.mp4`, `i2.mp4`, ...
- Place desired target reports under `data/outputs/` as `o1doc.docx` and/or `o1tex.tex`, `o2doc.docx`, ...
- The script will: extract audio, transcribe with faster-whisper, and pair each `iN.*` with the corresponding `oN*` target (prefers `.docx` if both exist). Temporary artifacts go to `outputs/finetune_tmp/`.

### Training command examples
JSON-based:
```powershell
python -m src.ai_lecture_summarizer.finetune_lora --base_model google/gemma-2b-it --train_json data/history.json --output_dir outputs/lora_summarizer
```

From videos and targets:
```powershell
python -m src.ai_lecture_summarizer.finetune_lora --from_videos --input_videos_dir data/input --target_outputs_dir data/outputs --tmp_dir outputs/finetune_tmp --whisper_size base --output_dir outputs/lora_summarizer
```

Key arguments:
- `--batch_size` (default 1), `--grad_accum` (default 8) → effective batch size
- `--lr` (default 2e-4), `--epochs` (default 3)
- `--max_length` (default 3072 tokens)
- LoRA config: `--lora_r` (8), `--lora_alpha` (16), `--lora_dropout` (0.05)
- Split: `--eval_split` (0.2), `--seed` (42)

Under the hood:
- Model loading via `transformers` with automatic device mapping and BF16/FP16 when CUDA is available
- LoRA with `peft` targeting proj and MLP modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Trainer with `DataCollatorForLanguageModeling(mlm=False)`

### Evaluation metrics
After training, the script evaluates on the held-out split and writes `report.json` and `report.txt` in `<output_dir>/eval/` with:
- ROUGE-1/2/L (precision/recall/F1)
- BLEU
- Accuracy@ROUGE1-F1≥0.5
- Structure coverage (presence of expected section headers)

### Using the adapter for inference
Set environment variables so the summarizer prefers the local adapter:
```powershell
$env:SUMMARIZER_MODE = "lora"
$env:LORA_ADAPTER_DIR = "outputs/lora_summarizer"
$env:LOCAL_MODEL_NAME = "google/gemma-2b-it"
```
Then run the CLI as usual. The pipeline will load the base model + adapter and generate locally.

---

## Web Interface (optional)
A minimal Node/Express server with a static UI is provided.

Install and run:
```powershell
npm install
npm run start
```

Open `http://localhost:3000`, upload a video, select summarizer mode, optionally provide a Gemini API key, and click Summarize. When finished, you can download `.docx` and `.tex`. The server runs the Python pipeline as:
```powershell
python main.py --input-dir uploads --output-dir outputs --model-size base --gemini-model gemini-1.5-flash --web-single
```

Headers supported by the server (`server/index.js`):
- `x-gemini-key`: per-request Gemini API key (if not set on the server)
- `x-summarizer-mode`: `lora` or `gemini`

---

## Data and Outputs
- Inputs: discovered recursively under `--input-dir` (extensions: `.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`)
- Outputs (within `--output-dir`):
  - `audio/`: `<video>.wav`
  - `transcripts/`: `<video>.txt`
  - `reports/`: `<video>_summary.docx`
  - `latex/`: `<video>_summary.tex`
  - `pdf/`: `<video>_summary.pdf` (if compiled)
  - `summary.docx`, `summary.tex`, `summary.json` for web usage (`--web-single`). `summary.json` contains a `bullets` array extracted from the final summary.

---

## Troubleshooting
- ffmpeg not found: ensure it is installed and on PATH (`video.py` checks this).
- pdflatex not found: install a LaTeX distribution or skip `--pdf`.
- LoRA mode errors: verify `LORA_ADAPTER_DIR` points to a saved adapter directory and that the base model matches the one used for training.
- Gemini auth errors: set `GEMINI_API_KEY` or pass `--api-key` / `x-gemini-key`.

---

## License
Specify your license here (e.g., MIT). If omitted, the project is proprietary by default.

