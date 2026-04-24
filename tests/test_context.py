"""Tests for cross-language source context retrieval."""

from __future__ import annotations

from agentbible.context import ContextManager


class TestContextManager:
    """Tests for source-tree retrieval."""

    def test_language_filters_hit_expected_trees(self) -> None:
        manager = ContextManager()
        assert ".py" in manager.build_context(query="provenance", lang="python", top_k=3)
        assert ".hpp" in manager.build_context(query="unitary", lang="cpp", top_k=3)
        assert ".rs" in manager.build_context(query="unitary", lang="rust", top_k=3)
        assert ".jl" in manager.build_context(query="unitary", lang="julia", top_k=3)

    def test_build_context_respects_language_filter(self) -> None:
        manager = ContextManager()
        context = manager.build_context(query="unitary", lang="rust", top_k=3)
        assert "rust" in context
        assert ".rs" in context
