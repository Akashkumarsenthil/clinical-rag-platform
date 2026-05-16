"""Metadata search endpoint for filtering documents by clinical attributes."""

from __future__ import annotations

from datetime import date
from typing import Optional

import structlog
from dateutil import parser as dateutil_parser
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import and_, or_

from src.api.models_v2 import SearchResult
from src.db.engine import get_session
from src.db.models import DocumentMetadata, DocumentRecord

router = APIRouter(tags=["Search"])
logger = structlog.get_logger(__name__)


def _parse_date(value: str) -> date:
    """Parse a natural-language date string into an ISO date.

    Handles phrasings like "Mar 8 2001", "2001-03-08", "March 8, 2001".

    Raises:
        ValueError: If the date string cannot be parsed.
    """
    return dateutil_parser.parse(value).date()


@router.get(
    "/search",
    response_model=list[SearchResult],
    summary="Search documents by clinical metadata",
)
async def search_documents(
    q: Optional[str] = Query(default=None, description="General text search across metadata fields."),
    mrn: Optional[str] = Query(default=None, description="Medical record number."),
    first_name: Optional[str] = Query(default=None, description="Patient first name (case-insensitive)."),
    last_name: Optional[str] = Query(default=None, description="Patient last name (case-insensitive)."),
    name: Optional[str] = Query(default=None, description="Searches both first and last name (case-insensitive)."),
    dob: Optional[str] = Query(default=None, description="Date of birth (natural language, e.g. 'Mar 8 2001')."),
    age: Optional[int] = Query(default=None, ge=0, description="Patient age."),
    document_type: Optional[str] = Query(default=None, description="Clinical document type."),
    date_from: Optional[str] = Query(default=None, description="Encounter date lower bound (inclusive)."),
    date_to: Optional[str] = Query(default=None, description="Encounter date upper bound (inclusive)."),
) -> list[SearchResult]:
    """Search documents by clinical metadata attributes.

    All filters are combined with AND logic.  The ``name`` parameter is a
    convenience that searches both ``first_name`` and ``last_name`` columns.
    Dates can be provided in natural language (e.g. "Mar 8 2001") and are
    normalised to ISO format automatically.

    Returns:
        List of matching documents with key metadata preview fields.

    Raises:
        HTTPException 400: If a date parameter cannot be parsed.
    """
    logger.info(
        "search_start",
        q=q,
        mrn=mrn,
        name=name,
        dob=dob,
        document_type=document_type,
    )

    filters = []

    if mrn is not None:
        filters.append(DocumentMetadata.mrn == mrn)

    if first_name is not None:
        filters.append(DocumentMetadata.first_name.ilike(f"%{first_name}%"))

    if last_name is not None:
        filters.append(DocumentMetadata.last_name.ilike(f"%{last_name}%"))

    if name is not None:
        filters.append(
            or_(
                DocumentMetadata.first_name.ilike(f"%{name}%"),
                DocumentMetadata.last_name.ilike(f"%{name}%"),
            )
        )

    if dob is not None:
        try:
            parsed_dob = _parse_date(dob)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid dob format: {exc}") from exc
        filters.append(DocumentMetadata.dob == parsed_dob)

    if age is not None:
        filters.append(DocumentMetadata.age == age)

    if document_type is not None:
        filters.append(DocumentMetadata.document_type.ilike(f"%{document_type}%"))

    if date_from is not None:
        try:
            parsed_from = _parse_date(date_from)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date_from format: {exc}") from exc
        filters.append(DocumentMetadata.encounter_date >= parsed_from)

    if date_to is not None:
        try:
            parsed_to = _parse_date(date_to)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date_to format: {exc}") from exc
        filters.append(DocumentMetadata.encounter_date <= parsed_to)

    if q is not None:
        q_pattern = f"%{q}%"
        filters.append(
            or_(
                DocumentMetadata.patient_name.ilike(q_pattern),
                DocumentMetadata.mrn.ilike(q_pattern),
                DocumentMetadata.document_type.ilike(q_pattern),
                DocumentMetadata.provider.ilike(q_pattern),
            )
        )

    session = get_session()
    try:
        query = (
            session.query(DocumentRecord, DocumentMetadata)
            .join(DocumentMetadata, DocumentRecord.doc_id == DocumentMetadata.doc_id)
        )

        if filters:
            query = query.filter(and_(*filters))

        query = query.order_by(DocumentRecord.uploaded_at.desc())
        rows = query.all()

        results = [
            SearchResult(
                doc_id=record.doc_id,
                filename=record.filename,
                status=record.status.value,
                chunk_count=record.chunk_count,
                patient_name=meta.patient_name,
                mrn=meta.mrn,
                dob=meta.dob,
                document_type=meta.document_type,
            )
            for record, meta in rows
        ]

        logger.info("search_done", results=len(results))
        return results
    finally:
        session.close()
