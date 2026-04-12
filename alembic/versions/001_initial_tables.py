"""Initial document management tables.

Revision ID: 001
Revises: None
Create Date: 2026-06-01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(36), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=True),
        sa.Column("file_hash", sa.String(64), unique=True, nullable=True),
        sa.Column(
            "status",
            sa.Enum("QUEUED", "EXTRACTING", "SUMMARIZING", "EMBEDDING", "COMPLETED", "FAILED", name="documentstatus"),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("uploaded_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("error_msg", sa.Text, nullable=True),
    )

    op.create_table(
        "document_metadata",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.doc_id"), unique=True, nullable=False),
        sa.Column("patient_name", sa.String(256), nullable=True),
        sa.Column("first_name", sa.String(128), nullable=True),
        sa.Column("last_name", sa.String(128), nullable=True),
        sa.Column("dob", sa.Date, nullable=True),
        sa.Column("age", sa.Integer, nullable=True),
        sa.Column("sex", sa.String(20), nullable=True),
        sa.Column("mrn", sa.String(64), nullable=True),
        sa.Column("encounter_date", sa.Date, nullable=True),
        sa.Column("provider", sa.String(256), nullable=True),
        sa.Column("document_type", sa.String(128), nullable=True),
        sa.Column("raw_json", sa.JSON, nullable=True),
    )

    op.create_table(
        "document_summaries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.doc_id"), unique=True, nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("document_summaries")
    op.drop_table("document_metadata")
    op.drop_table("documents")
    op.execute("DROP TYPE IF EXISTS documentstatus")
