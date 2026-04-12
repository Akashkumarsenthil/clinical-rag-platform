"""SQLAlchemy ORM models for document management."""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class DocumentStatus(str, enum.Enum):
    """Processing stages for uploaded documents."""

    QUEUED = "QUEUED"
    EXTRACTING = "EXTRACTING"
    SUMMARIZING = "SUMMARIZING"
    EMBEDDING = "EMBEDDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentRecord(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(1024))
    file_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.QUEUED, nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(),
    )
    error_msg: Mapped[str | None] = mapped_column(Text)

    metadata_rel: Mapped[DocumentMetadata | None] = relationship(
        back_populates="document", uselist=False, cascade="all, delete-orphan",
    )
    summary_rel: Mapped[DocumentSummary | None] = relationship(
        back_populates="document", uselist=False, cascade="all, delete-orphan",
    )


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.doc_id"), unique=True, nullable=False,
    )
    patient_name: Mapped[str | None] = mapped_column(String(256))
    first_name: Mapped[str | None] = mapped_column(String(128))
    last_name: Mapped[str | None] = mapped_column(String(128))
    dob: Mapped[date | None] = mapped_column(Date)
    age: Mapped[int | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String(20))
    mrn: Mapped[str | None] = mapped_column(String(64))
    encounter_date: Mapped[date | None] = mapped_column(Date)
    provider: Mapped[str | None] = mapped_column(String(256))
    document_type: Mapped[str | None] = mapped_column(String(128))
    raw_json: Mapped[dict | None] = mapped_column(JSON)

    document: Mapped[DocumentRecord] = relationship(back_populates="metadata_rel")


class DocumentSummary(Base):
    __tablename__ = "document_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.doc_id"), unique=True, nullable=False,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(),
    )

    document: Mapped[DocumentRecord] = relationship(back_populates="summary_rel")
