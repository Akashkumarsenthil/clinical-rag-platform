"""Unit tests for document API endpoints with mocked dependencies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.routes.documents import _compute_percent


class TestComputePercent:
    """Verify the progress percentage derivation logic."""

    def test_stage0_returns_zero(self):
        assert _compute_percent("unknown", 0, 0, 0) == 0.0

    def test_stage1_returns_33(self):
        assert _compute_percent("extracting_metadata", 1, 0, 0) == 33.0

    def test_stage2_returns_66(self):
        assert _compute_percent("generating_summary", 2, 0, 0) == 66.0

    def test_stage3_partial(self):
        pct = _compute_percent("embedding_chunks", 3, 256, 512)
        assert pct == pytest.approx(83.0, abs=0.1)

    def test_stage3_complete(self):
        pct = _compute_percent("embedding_chunks", 3, 512, 512)
        assert pct == pytest.approx(100.0, abs=0.1)

    def test_stage3_zero_total(self):
        pct = _compute_percent("embedding_chunks", 3, 0, 0)
        assert pct == 66.0
