"""Integration tests for the IngestionPipeline using mocked external services."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pipeline import IngestionPipeline, IngestionResult, PipelineError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pdf_pages(doc_id: str = "doc-0001", file_hash: str = "abc123") -> list:
    """Return mock Document objects simulating a 2-page PDF."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Metformin is used to treat type 2 diabetes mellitus.",
            metadata={
                "source": "/tmp/test.pdf",
                "page_number": 1,
                "total_pages": 2,
                "section_header": "Introduction",
                "doc_id": doc_id,
                "file_hash": file_hash,
            },
        ),
        Document(
            page_content="The typical dose is 500 mg twice daily with meals.",
            metadata={
                "source": "/tmp/test.pdf",
                "page_number": 2,
                "total_pages": 2,
                "section_header": "Dosing",
                "doc_id": doc_id,
                "file_hash": file_hash,
            },
        ),
    ]


def _make_pipeline(
    qdrant_mock: MagicMock,
    pdf_pages: list | None = None,
    already_ingested: bool = False,
    file_hash: str = "abc123",
) -> IngestionPipeline:
    """Build an IngestionPipeline with all external dependencies mocked."""
    embedder_mock = MagicMock()
    embedder_mock.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._qdrant = qdrant_mock
    pipeline._embedder = embedder_mock
    pipeline._strategy = "recursive"
    pipeline._collection = "clinical_docs"

    loader_mock = MagicMock()
    loader_mock.load.return_value = pdf_pages or _make_mock_pdf_pages(file_hash=file_hash)
    pipeline._loader = loader_mock

    if already_ingested:
        # Simulate a scroll that returns existing points
        fake_point = MagicMock()
        qdrant_mock.scroll.return_value = ([fake_point], None)
    else:
        qdrant_mock.scroll.return_value = ([], None)

    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestionPipeline:
    def test_full_pipeline_run_success(self, mock_qdrant_client, tmp_path):
        """Full pipeline run with mocked services should return success result."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        pipeline = _make_pipeline(mock_qdrant_client)

        # Patch chunker to return the pages as-is (each page = one chunk)
        with patch("src.ingestion.pipeline.get_chunker") as mock_get_chunker:
            chunker_mock = MagicMock()
            chunker_mock.chunk.return_value = _make_mock_pdf_pages()
            mock_get_chunker.return_value = chunker_mock

            result = pipeline.run(str(pdf_path))

        assert isinstance(result, IngestionResult)
        assert result.status == "success"
        assert result.chunks_created == 2
        assert result.skipped is False
        assert result.doc_id == "doc-0001"

    def test_idempotency_second_run_is_skipped(self, mock_qdrant_client, tmp_path):
        """Running the pipeline twice with the same file should skip on second run."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        pipeline = _make_pipeline(
            mock_qdrant_client,
            already_ingested=True,
            file_hash="abc123",
        )

        result = pipeline.run(str(pdf_path))

        assert result.skipped is True
        assert result.status == "skipped"
        assert result.chunks_created == 0
        # Upsert should NOT have been called
        mock_qdrant_client.upsert.assert_not_called()

    def test_invalid_path_raises_pipeline_error(self, mock_qdrant_client):
        """Non-existent file path should raise PipelineError."""
        pipeline = _make_pipeline(mock_qdrant_client)
        # Override loader to raise as PDFLoader would for missing file
        pipeline._loader.load.side_effect = Exception("File not found: /no/such/file.pdf")

        with pytest.raises(PipelineError, match="PDF loading failed"):
            pipeline.run("/no/such/file.pdf")

    def test_qdrant_upsert_is_called_with_correct_collection(
        self, mock_qdrant_client, tmp_path
    ):
        """Upsert should target the configured collection name."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        pipeline = _make_pipeline(mock_qdrant_client)
        with patch("src.ingestion.pipeline.get_chunker") as mock_get_chunker:
            chunker_mock = MagicMock()
            chunker_mock.chunk.return_value = _make_mock_pdf_pages()
            mock_get_chunker.return_value = chunker_mock
            pipeline.run(str(pdf_path))

        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "clinical_docs"

    def test_embedding_failure_raises_pipeline_error(self, mock_qdrant_client, tmp_path):
        """Embedding failure should propagate as PipelineError."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        pipeline = _make_pipeline(mock_qdrant_client)
        pipeline._embedder.embed_documents.side_effect = Exception("OpenAI rate limit")

        with patch("src.ingestion.pipeline.get_chunker") as mock_get_chunker:
            chunker_mock = MagicMock()
            chunker_mock.chunk.return_value = _make_mock_pdf_pages()
            mock_get_chunker.return_value = chunker_mock

            with pytest.raises(PipelineError, match="Embedding failed"):
                pipeline.run(str(pdf_path))
