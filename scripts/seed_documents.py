#!/usr/bin/env python3
"""Seed script: run the sample PDFs through the full 3-stage pipeline.

Usage (from container or venv):
    python scripts/seed_documents.py

This processes all PDFs in data/sample_pdfs/ through metadata extraction,
summary generation, and embedding. The app must have database and Qdrant
connections available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_loader import PDFLoader
from src.db.engine import get_session
from src.db.models import Base, DocumentRecord, DocumentStatus
from src.db.engine import engine
from src.services.document_pipeline import DocumentPipeline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_pdfs"


def main() -> None:
    Base.metadata.create_all(bind=engine)

    if not SAMPLE_DIR.exists():
        print(f"Sample directory not found: {SAMPLE_DIR}")
        return

    pdfs = sorted(SAMPLE_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {SAMPLE_DIR}")
        return

    print(f"Found {len(pdfs)} PDF(s) to process")
    pipeline = DocumentPipeline()

    for pdf_path in pdfs:
        data = pdf_path.read_bytes()
        doc_id, file_hash = PDFLoader.doc_id_from_bytes(data)

        session = get_session()
        try:
            existing = session.get(DocumentRecord, doc_id)
            if existing and existing.status == DocumentStatus.COMPLETED:
                print(f"  SKIP {pdf_path.name} (already completed, doc_id={doc_id[:12]}...)")
                continue

            storage_path = pdf_path
            if not existing:
                record = DocumentRecord(
                    doc_id=doc_id,
                    filename=pdf_path.name,
                    storage_path=str(storage_path),
                    file_hash=file_hash,
                    status=DocumentStatus.QUEUED,
                )
                session.add(record)
            else:
                existing.status = DocumentStatus.QUEUED
                existing.error_msg = None

            session.commit()
        finally:
            session.close()

        print(f"  Processing {pdf_path.name} (doc_id={doc_id[:12]}...)...")

        try:
            pipeline.run(doc_id, str(storage_path))
            print(f"    DONE")
        except Exception as exc:
            print(f"    FAILED: {exc}")

    print("Seed complete.")


if __name__ == "__main__":
    main()
