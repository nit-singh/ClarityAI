from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

LATEX_PREAMBLE = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{titlesec}
\usepackage{titling}
\usepackage{setspace}
\setstretch{1.15}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
\begin{document}
"""

LATEX_END = "\n\\end{document}\n"


def _latex_escape(text: str) -> str:
	return (
		text.replace("\\", r"\textbackslash{}")
		.replace("%", r"\%")
		.replace("$", r"\$")
		.replace("#", r"\#")
		.replace("_", r"\_")
		.replace("{", r"\{")
		.replace("}", r"\}")
		.replace("~", r"\textasciitilde{}")
		.replace("^", r"\textasciicircum{}")
	)


def _title_page(title: str, subtitle: str | None) -> str:
	parts = ["\\begin{center}", f"\\LARGE\\textbf{{{_latex_escape(title)}}}\\\\[1em]"]
	if subtitle:
		parts.append(f"\\normalsize {_latex_escape(subtitle)}\\\\")
	parts.append("\\end{center}\n\\vspace{1em}\n\\hrule\\vspace{1em}")
	return "\n".join(parts)


def _open_itemize() -> str:
	return "\\begin{itemize}[leftmargin=*]"


def _close_itemize() -> str:
	return "\\end{itemize}"


def _parse_summary_to_latex(summary: str) -> str:
	lines = [l.rstrip() for l in summary.splitlines()]
	out_lines: list[str] = []
	in_list = False
	paragraph_buf: list[str] = []

	def flush_paragraph():
		if paragraph_buf:
			out_lines.append(_latex_escape(" ".join(paragraph_buf).strip()))
			out_lines.append("")
			paragraph_buf.clear()

	for raw in lines:
		line = raw.strip()
		if not line:
			flush_paragraph()
			continue
		if line.startswith("## "):
			if in_list:
				out_lines.append(_close_itemize())
				in_list = False
			flush_paragraph()
			out_lines.append(f"\\section{{{_latex_escape(line[3:].strip())}}}")
			continue
		if line.startswith("- ") or line.startswith("* "):
			if not in_list:
				flush_paragraph()
				out_lines.append(_open_itemize())
				in_list = True
			out_lines.append(f"\\item {_latex_escape(line[2:].strip())}")
			continue
		paragraph_buf.append(line)

	flush_paragraph()
	if in_list:
		out_lines.append(_close_itemize())

	return "\n".join(out_lines)


def build_latex_report(summary_text: str, output_path: Path, title: str, source_file: Path | None = None) -> Path:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	subtitle = None
	if source_file is not None:
		subtitle = f"Source: {source_file.name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

	content = []
	content.append(LATEX_PREAMBLE)
	content.append(_title_page(title, subtitle))
	content.append(_parse_summary_to_latex(summary_text))
	content.append(LATEX_END)
	output_path.write_text("\n".join(content), encoding="utf-8")
	return output_path


def compile_pdf(latex_file: Path, work_dir: Path | None = None) -> Path:
	if shutil.which("pdflatex") is None:
		raise RuntimeError("pdflatex not found on PATH. Install TeX Live or MiKTeX.")
	latex_file = Path(latex_file)
	cwd = Path(work_dir) if work_dir else latex_file.parent
	cmd = ["pdflatex", "-interaction=nonstopmode", latex_file.name]
	subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	pdf_path = latex_file.with_suffix(".pdf")
	return pdf_path
