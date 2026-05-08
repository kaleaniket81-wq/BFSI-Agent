"""ingestion/src/parser/document_parser.py — Extract text from PDF and DOCX files."""
import fitz          # PyMuPDF
import docx as pydocx
from pathlib import Path
from typing import Dict, List


class DocumentParser:
    """Parse PDF and DOCX files into page-level text."""

    SUPPORTED = {".pdf", ".docx", ".doc"}

    def parse(self, filepath: str) -> Dict:
        path = Path(filepath)
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(filepath)
        else:
            return self._parse_docx(filepath)

    # ── PDF ──────────────────────────────────────────────────────
    def _parse_pdf(self, filepath: str) -> Dict:
        doc = fitz.open(filepath)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({"page_number": i + 1, "text": text})
        return {
            "filename":   Path(filepath).name,
            "file_type":  "pdf",
            "page_count": len(doc),
            "pages":      pages,
        }

    # ── DOCX ─────────────────────────────────────────────────────
    def _parse_docx(self, filepath: str) -> Dict:
        doc = pydocx.Document(filepath)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        # Treat each 50-line block as a "page"
        lines = full_text.split("\n")
        pages, chunk_size = [], 50
        for i in range(0, max(len(lines), 1), chunk_size):
            block = "\n".join(lines[i:i + chunk_size]).strip()
            if block:
                pages.append({"page_number": i // chunk_size + 1, "text": block})
        return {
            "filename":   Path(filepath).name,
            "file_type":  "docx",
            "page_count": len(pages),
            "pages":      pages,
        }
