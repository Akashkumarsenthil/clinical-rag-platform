"""PDF document loader using PyMuPDF (fitz)."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import structlog
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

TITLE_PATTERN = re.compile(
    r"^(?:Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References?|"
    r"Background|Related Work|Appendix|Section \d+|Chapter \d+|\d+\.?\s+[A-Z])",
    re.IGNORECASE,
)


class PDFLoadError(Exception):
    """Raised when a PDF file cannot be loaded or parsed."""


class PDFLoader:
    """Loads PDF files into LangChain Document objects using PyMuPDF.

    Each page is returned as a separate Document with metadata including
    source path, page number, total pages, inferred section header, a
    unique document ID, and a SHA-256 hash of the file content.
    """

    def load(self, path: str) -> list[Document]:
        """Load a PDF file and return a list of Documents, one per page.

        Args:
            path: Filesystem path to the PDF file.

        Returns:
            List of LangChain Document objects with populated metadata.

        Raises:
            PDFLoadError: If the file cannot be opened or parsed.
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise PDFLoadError(f"File not found: {resolved}")

        file_hash = self._compute_hash(resolved)
        doc_id = self.doc_id_from_hash(file_hash)

        logger.info("loading_pdf", path=str(resolved), doc_id=doc_id, file_hash=file_hash)

        try:
            pdf = fitz.open(str(resolved))
        except Exception as exc:
            raise PDFLoadError(f"Failed to open PDF '{resolved}': {exc}") from exc

        total_pages = len(pdf)
        documents: list[Document] = []
        current_section: str = ""

        for page_num in range(total_pages):
            page = pdf[page_num]
            text = page.get_text("text")

            # Detect section header from bold spans or title-pattern lines
            section_header = self._extract_section_header(page, text, current_section)
            if section_header:
                current_section = section_header

            metadata: dict[str, Any] = {
                "source": str(resolved),
                "page_number": page_num + 1,
                "total_pages": total_pages,
                "section_header": current_section,
                "doc_id": doc_id,
                "file_hash": file_hash,
            }
            documents.append(Document(page_content=text, metadata=metadata))

        pdf.close()
        logger.info(
            "pdf_loaded",
            doc_id=doc_id,
            total_pages=total_pages,
            pages_returned=len(documents),
        )
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def doc_id_from_hash(file_hash: str) -> str:
        """Derive a deterministic doc_id from a SHA-256 file hash.

        Same bytes always produce the same doc_id regardless of file path.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, file_hash))

    @staticmethod
    def doc_id_from_bytes(data: bytes) -> tuple[str, str]:
        """Compute (doc_id, file_hash) from raw file bytes."""
        file_hash = hashlib.sha256(data).hexdigest()
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_hash))
        return doc_id, file_hash

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Return the SHA-256 hex digest of the file at *path*."""
        sha256 = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _extract_section_header(
        page: fitz.Page,
        text: str,
        current_section: str,
    ) -> str:
        """Attempt to detect the section heading for this page.

        Strategy:
        1. Walk text spans; if a span is bold and ≥ 4 characters, treat it as a heading.
        2. Fall back to scanning the first few lines with the TITLE_PATTERN regex.

        Returns the detected heading or the existing *current_section* if none found.
        """
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") != 0:  # 0 == text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & 2**4)  # bit 4 = bold
                        span_text = span.get("text", "").strip()
                        if is_bold and len(span_text) >= 4:
                            return span_text[:120]
        except Exception:
            pass  # gracefully degrade to regex approach

        for line in text.splitlines()[:10]:
            line = line.strip()
            if TITLE_PATTERN.match(line) and 4 <= len(line) <= 120:
                return line

        return current_section
