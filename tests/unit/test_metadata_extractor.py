"""Unit tests for MetadataExtractor with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.services.metadata_extractor import MetadataExtractor, ExtractedMetadata


class TestExtractedMetadata:
    """Validation of the pydantic model."""

    def test_normalize_dob_iso(self):
        m = ExtractedMetadata(dob="2001-03-08")
        assert m.dob == "2001-03-08"

    def test_normalize_dob_natural(self):
        m = ExtractedMetadata(dob="Mar 8, 2001")
        assert m.dob == "2001-03-08"

    def test_normalize_dob_null(self):
        m = ExtractedMetadata(dob=None)
        assert m.dob is None

    def test_normalize_dob_invalid(self):
        m = ExtractedMetadata(dob="not-a-date")
        assert m.dob is None

    def test_coerce_age_int(self):
        m = ExtractedMetadata(age=45)
        assert m.age == 45

    def test_coerce_age_string(self):
        m = ExtractedMetadata(age="45")
        assert m.age == 45

    def test_coerce_age_none(self):
        m = ExtractedMetadata(age=None)
        assert m.age is None


class TestMetadataExtractor:
    """Tests for the full extraction pipeline with mocked LLM."""

    @patch("src.services.metadata_extractor._get_llm")
    def test_extract_parses_json(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''{
            "patient_name": "John Smith",
            "first_name": "John",
            "last_name": "Smith",
            "dob": "1990-01-15",
            "age": 36,
            "sex": "Male",
            "mrn": "123456",
            "encounter_date": "2026-06-01",
            "provider": "Dr. Jones",
            "document_type": "Discharge Summary"
        }'''
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        extractor = MetadataExtractor()
        result = extractor.extract("Sample clinical document text")

        assert result["patient_name"] == "John Smith"
        assert result["mrn"] == "123456"
        assert result["dob"] == "1990-01-15"
        assert result["age"] == 36

    @patch("src.services.metadata_extractor._get_llm")
    def test_extract_handles_markdown_fences(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"patient_name": "Jane Doe", "mrn": "789012"}\n```'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        extractor = MetadataExtractor()
        result = extractor.extract("Some text")

        assert result["patient_name"] == "Jane Doe"
        assert result["mrn"] == "789012"

    @patch("src.services.metadata_extractor._get_llm")
    def test_regex_fallback_for_mrn(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"patient_name": "John"}'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        extractor = MetadataExtractor()
        text = "Patient Name: John\nMRN: 12345678\nDOB: March 15, 1990"
        result = extractor.extract(text)

        assert result["mrn"] == "12345678"

    @patch("src.services.metadata_extractor._get_llm")
    def test_llm_failure_falls_back_to_regex(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        mock_get_llm.return_value = mock_llm

        extractor = MetadataExtractor()
        text = "MRN: 99887766\nDate of Birth: 2001-03-08"
        result = extractor.extract(text)

        assert result["mrn"] == "99887766"


class TestDocIdPolicy:
    """Verify the file-hash-based doc_id policy."""

    def test_same_bytes_same_docid(self):
        from src.ingestion.pdf_loader import PDFLoader

        data = b"fake pdf content for testing"
        doc_id_1, hash_1 = PDFLoader.doc_id_from_bytes(data)
        doc_id_2, hash_2 = PDFLoader.doc_id_from_bytes(data)

        assert doc_id_1 == doc_id_2
        assert hash_1 == hash_2

    def test_different_bytes_different_docid(self):
        from src.ingestion.pdf_loader import PDFLoader

        doc_id_1, _ = PDFLoader.doc_id_from_bytes(b"content A")
        doc_id_2, _ = PDFLoader.doc_id_from_bytes(b"content B")

        assert doc_id_1 != doc_id_2
