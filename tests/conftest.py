"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.dense_retriever import ScoredChunk


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents() -> list[Document]:
    """Return 5 realistic clinical Document objects for use in tests."""
    clinical_texts = [
        (
            "Metformin is the first-line pharmacological therapy for type 2 diabetes mellitus. "
            "It reduces hepatic glucose production through activation of AMP-activated protein kinase "
            "(AMPK). The typical starting dose is 500 mg twice daily with meals, titrated to a maximum "
            "of 2000 mg per day. Gastrointestinal side effects are the most common adverse events and "
            "can be minimised by gradual dose titration and administration with food."
        ),
        (
            "Acute myocardial infarction (AMI) requires immediate reperfusion therapy. "
            "Primary percutaneous coronary intervention (PCI) is preferred over fibrinolysis when "
            "performed within 90 minutes of first medical contact. Dual antiplatelet therapy with "
            "aspirin and a P2Y12 inhibitor (ticagrelor or prasugrel) should be initiated as soon as "
            "STEMI is diagnosed. High-intensity statin therapy is recommended for all AMI patients."
        ),
        (
            "Community-acquired pneumonia (CAP) severity should be assessed using validated scores "
            "such as CURB-65 or the Pneumonia Severity Index (PSI). Empirical antibiotic therapy "
            "for outpatient CAP without comorbidities consists of amoxicillin 1g TID for 5 days, "
            "or a macrolide where atypical pathogens are suspected. Inpatient CAP treatment typically "
            "combines a beta-lactam with a macrolide or respiratory fluoroquinolone monotherapy."
        ),
        (
            "Chronic kidney disease (CKD) staging is based on eGFR and albuminuria categories. "
            "CKD stage 3a is defined as eGFR 45-59 mL/min/1.73m². Key management goals include "
            "blood pressure control to <130/80 mmHg using ACE inhibitors or ARBs in patients with "
            "albuminuria, glycaemic control in diabetic CKD, and avoidance of nephrotoxic agents. "
            "SGLT2 inhibitors have demonstrated reno-protective effects independent of glucose lowering."
        ),
        (
            "Rheumatoid arthritis (RA) management follows a treat-to-target strategy aiming for "
            "remission or low disease activity. Methotrexate (MTX) remains the anchor DMARD, typically "
            "started at 10-15 mg weekly and escalated to 20-25 mg as tolerated. Folic acid 5 mg "
            "weekly is co-prescribed to reduce MTX-related adverse effects. Biologic DMARDs including "
            "TNF inhibitors, IL-6 receptor antagonists, and JAK inhibitors are added when adequate "
            "response to conventional DMARDs is not achieved after 3-6 months."
        ),
    ]

    documents: list[Document] = []
    for i, text in enumerate(clinical_texts):
        doc = Document(
            page_content=text,
            metadata={
                "source": f"/data/clinical_doc_{i+1}.pdf",
                "page_number": 1,
                "total_pages": 5,
                "section_header": ["Introduction", "Diagnosis", "Treatment", "Management", "Pharmacology"][i],
                "doc_id": f"doc-{i+1:04d}",
                "file_hash": f"abc{i}def{i}123{i}" * 4,
            },
        )
        documents.append(doc)
    return documents


@pytest.fixture
def sample_scored_chunks(sample_documents) -> list[ScoredChunk]:
    """Return ScoredChunk objects derived from sample_documents."""
    return [
        ScoredChunk(
            content=doc.page_content,
            metadata=doc.metadata,
            score=0.9 - idx * 0.05,
        )
        for idx, doc in enumerate(sample_documents)
    ]


# ---------------------------------------------------------------------------
# Service mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_qdrant_client():
    """Return a MagicMock simulating a QdrantClient."""
    mock = MagicMock()
    # get_collections returns an object with a .collections attribute
    mock.get_collections.return_value.collections = []
    # scroll returns (points, next_page_offset)
    mock.scroll.return_value = ([], None)
    # search returns empty list by default
    mock.search.return_value = []
    return mock


@pytest.fixture
def mock_redis_client():
    """Return a MagicMock simulating a Redis client."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.incr.return_value = 1
    return mock


@pytest.fixture
def mock_openai_client():
    """Return a MagicMock simulating an OpenAI client."""
    mock = MagicMock()

    # Embeddings
    embedding_data = MagicMock()
    embedding_data.embedding = [0.1] * 1536
    mock.embeddings.create.return_value.data = [embedding_data]

    # Chat completions
    message = MagicMock()
    message.content = "This is a mock answer from GPT-4o."
    mock.chat.completions.create.return_value.choices = [MagicMock(message=message)]

    return mock
