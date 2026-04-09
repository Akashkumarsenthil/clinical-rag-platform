#!/usr/bin/env python3
"""Download and ingest sample open-access clinical/medical PDFs from arXiv."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import httpx
import structlog

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import IngestionPipeline, PipelineError

log = structlog.get_logger(__name__)

# Public-domain / open-access medical papers (arXiv)
SAMPLE_URLS: list[tuple[str, str]] = [
    (
        "https://arxiv.org/pdf/2402.01821",
        "clinical_nlp_survey_2024.pdf",
    ),
    (
        "https://arxiv.org/pdf/2301.04246",
        "med_palm_large_language_models_medicine.pdf",
    ),
    (
        "https://arxiv.org/pdf/2310.00041",
        "clinical_text_mining_overview.pdf",
    ),
    (
        "https://arxiv.org/pdf/2109.07958",
        "biomedical_ner_survey.pdf",
    ),
    (
        "https://arxiv.org/pdf/2212.13138",
        "chatgpt_medical_knowledge.pdf",
    ),
]


def download_pdf(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a PDF from *url* to *dest*.

    Returns True on success, False on failure.
    """
    log.info("downloading_pdf", url=url, dest=str(dest))
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_bytes(chunk_size=65536):
                    fh.write(chunk)
        log.info("download_complete", dest=str(dest), size_bytes=dest.stat().st_size)
        return True
    except Exception as exc:
        log.error("download_failed", url=url, error=str(exc))
        return False


def main() -> None:
    """Download sample PDFs and run the ingestion pipeline on each."""
    pipeline = IngestionPipeline(chunk_strategy="recursive")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        results_summary: list[dict] = []

        for url, filename in SAMPLE_URLS:
            dest = tmp / filename
            if not download_pdf(url, dest):
                results_summary.append(
                    {"file": filename, "status": "download_failed", "chunks": 0}
                )
                continue

            try:
                result = pipeline.run(str(dest))
                results_summary.append(
                    {
                        "file": filename,
                        "doc_id": result.doc_id,
                        "status": result.status,
                        "chunks": result.chunks_created,
                        "skipped": result.skipped,
                    }
                )
                log.info(
                    "ingestion_result",
                    file=filename,
                    doc_id=result.doc_id,
                    chunks=result.chunks_created,
                    status=result.status,
                )
            except PipelineError as exc:
                log.error("ingestion_failed", file=filename, error=str(exc))
                results_summary.append(
                    {"file": filename, "status": "pipeline_error", "chunks": 0}
                )

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'File':<45} {'Status':<12} {'Chunks':>6}")
    print("-" * 70)
    for r in results_summary:
        print(f"{r['file']:<45} {r['status']:<12} {r.get('chunks', 0):>6}")
    print("=" * 70)
    total_chunks = sum(r.get("chunks", 0) for r in results_summary)
    print(f"Total chunks ingested: {total_chunks}")


if __name__ == "__main__":
    main()
