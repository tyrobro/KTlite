"""
test_app.py — Unit tests for the pure helper functions in app.py.

These tests cover build_context_block and build_sources without
importing Streamlit (they are pure functions). init_session_state is
tested by mocking st.session_state with a plain dict.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers — lightweight Document-like stub
# ---------------------------------------------------------------------------

def make_doc(page_content: str, source: str | None = "file.txt"):
    """Return a minimal object that mimics a LangChain Document."""
    metadata = {"source": source} if source is not None else {}
    return SimpleNamespace(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------

from app import build_context_block, build_sources, init_session_state


# ===========================================================================
# build_context_block
# ===========================================================================

class TestBuildContextBlock:
    def test_empty_list_returns_empty_string(self):
        assert build_context_block([]) == ""

    def test_single_doc_returns_its_content(self):
        doc = make_doc("Hello world")
        assert build_context_block([doc]) == "Hello world"

    def test_two_docs_joined_with_double_newline(self):
        docs = [make_doc("First"), make_doc("Second")]
        assert build_context_block(docs) == "First\n\nSecond"

    def test_three_docs_all_joined(self):
        docs = [make_doc("A"), make_doc("B"), make_doc("C")]
        assert build_context_block(docs) == "A\n\nB\n\nC"

    def test_preserves_internal_whitespace(self):
        docs = [make_doc("line1\nline2"), make_doc("line3")]
        assert build_context_block(docs) == "line1\nline2\n\nline3"

    def test_does_not_strip_content(self):
        docs = [make_doc("  padded  ")]
        assert build_context_block(docs) == "  padded  "


# ===========================================================================
# build_sources
# ===========================================================================

class TestBuildSources:
    def test_empty_list_returns_unknown_sentinel(self):
        assert build_sources([]) == ["Unknown"]

    def test_single_doc_with_source(self):
        doc = make_doc("content", source="report.pdf")
        assert build_sources([doc]) == ["report.pdf"]

    def test_missing_source_key_maps_to_unknown(self):
        doc = make_doc("content", source=None)  # metadata has no "source" key
        assert build_sources([doc]) == ["Unknown"]

    def test_duplicates_are_removed(self):
        docs = [
            make_doc("a", source="x.txt"),
            make_doc("b", source="x.txt"),
            make_doc("c", source="x.txt"),
        ]
        assert build_sources(docs) == ["x.txt"]

    def test_deduplication_preserves_insertion_order(self):
        docs = [
            make_doc("a", source="alpha.txt"),
            make_doc("b", source="beta.txt"),
            make_doc("c", source="alpha.txt"),
        ]
        result = build_sources(docs)
        assert result == ["alpha.txt", "beta.txt"]

    def test_all_distinct_sources_kept(self):
        docs = [
            make_doc("a", source="a.txt"),
            make_doc("b", source="b.txt"),
            make_doc("c", source="c.txt"),
        ]
        assert build_sources(docs) == ["a.txt", "b.txt", "c.txt"]

    def test_result_has_no_duplicates(self):
        docs = [make_doc(f"chunk{i}", source="same.pdf") for i in range(3)]
        result = build_sources(docs)
        assert len(result) == len(set(result))

    def test_result_is_never_empty(self):
        # Even with zero docs, result has at least one element
        assert len(build_sources([])) >= 1

    def test_mixed_known_and_missing_sources(self):
        docs = [
            make_doc("a", source="known.pdf"),
            make_doc("b", source=None),  # missing → "Unknown"
        ]
        result = build_sources(docs)
        assert "known.pdf" in result
        assert "Unknown" in result
        assert len(result) == 2


# ===========================================================================
# init_session_state
# ===========================================================================

class TestInitSessionState:
    def test_adds_messages_key_when_absent(self):
        fake_state = {}
        with patch("app.st") as mock_st:
            mock_st.session_state = fake_state
            init_session_state()
        assert "messages" in fake_state
        assert fake_state["messages"] == []

    def test_idempotent_does_not_overwrite_existing_messages(self):
        existing = [{"role": "user", "content": "hi"}]
        fake_state = {"messages": existing}
        with patch("app.st") as mock_st:
            mock_st.session_state = fake_state
            init_session_state()
        # List object must be the same reference — not reset
        assert fake_state["messages"] is existing

    def test_safe_to_call_multiple_times(self):
        fake_state = {}
        with patch("app.st") as mock_st:
            mock_st.session_state = fake_state
            init_session_state()
            init_session_state()
            init_session_state()
        assert fake_state["messages"] == []
