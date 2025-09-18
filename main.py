import sys
from pathlib import Path

# Ensure local src/ is importable when running from project root
CURRENT_DIR = Path(__file__).parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from ai_lecture_summarizer.cli import main

if __name__ == "__main__":
	main()
