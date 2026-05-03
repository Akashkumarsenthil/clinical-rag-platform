"""Metadata extraction from clinical PDF text using LLM + regex fallback.

NOTE: All patient data in this platform is synthetic/demo PHI generated
for testing purposes. No real patient information is processed.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Optional

import structlog
from pydantic import BaseModel, field_validator

from src.agents.nodes import _get_llm

logger = structlog.get_logger(__name__)

_EXTRACT_PROMPT = """You are a clinical document metadata extractor.
Extract the following fields from the clinical document text below.
Return ONLY a JSON object with these keys (use null for missing fields):

{{
  "patient_name": "full name",
  "first_name": "first name only",
  "last_name": "last name only",
  "dob": "YYYY-MM-DD",
  "age": integer or null,
  "sex": "Male/Female/Other",
  "mrn": "medical record number",
  "encounter_date": "YYYY-MM-DD",
  "provider": "provider/physician name",
  "document_type": "e.g. Discharge Summary, Progress Note, Lab Report"
}}

Document text (first 3000 chars):
{text}

JSON output:"""

# Regex patterns for fallback extraction
_MRN_PATTERN = re.compile(
    r"(?:MRN|MR#|Medical\s*Record\s*(?:Number|No\.?|#))\s*[:\s]*(\d{4,12})",
    re.IGNORECASE,
)
_DOB_PATTERN = re.compile(
    r"(?:DOB|Date\s*of\s*Birth|Birth\s*Date)\s*[:\s]*([A-Za-z0-9/\-,.\s]{6,20})",
    re.IGNORECASE,
)


class ExtractedMetadata(BaseModel):
    """Validated metadata fields extracted from a clinical document."""

    patient_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    mrn: Optional[str] = None
    encounter_date: Optional[str] = None
    provider: Optional[str] = None
    document_type: Optional[str] = None

    @field_validator("dob", "encounter_date", mode="before")
    @classmethod
    def normalize_date(cls, v: Any) -> Optional[str]:
        """Normalize date strings to ISO YYYY-MM-DD."""
        if not v or v == "null":
            return None
        try:
            from dateutil.parser import parse as parse_date
            return parse_date(str(v), fuzzy=True).date().isoformat()
        except Exception:
            return None

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> Optional[int]:
        if v is None or v == "null":
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class MetadataExtractor:
    """Extracts structured metadata from clinical document text."""

    def extract(self, text: str) -> dict[str, Any]:
        """Extract metadata fields from document text.

        Uses LLM extraction with regex fallback for MRN and DOB.
        """
        logger.info("metadata_extraction_start", text_len=len(text))
        truncated = text[:3000]

        try:
            llm = _get_llm()
            prompt = _EXTRACT_PROMPT.format(text=truncated)
            response = llm.invoke(prompt)
            raw = response.content.strip()  # type: ignore[union-attr]
            data = self._parse_json(raw)
        except Exception as exc:
            logger.warning("llm_extraction_failed", error=str(exc))
            data = {}

        data = self._regex_fallback(data, text)

        validated = ExtractedMetadata(**data)
        result = validated.model_dump()
        logger.info("metadata_extraction_done", fields_found=sum(1 for v in result.values() if v is not None))
        return result

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}

    @staticmethod
    def _regex_fallback(data: dict[str, Any], text: str) -> dict[str, Any]:
        """Fill in MRN and DOB via regex if the LLM missed them."""
        if not data.get("mrn"):
            m = _MRN_PATTERN.search(text)
            if m:
                data["mrn"] = m.group(1)

        if not data.get("dob"):
            m = _DOB_PATTERN.search(text)
            if m:
                data["dob"] = m.group(1).strip()

        return data
