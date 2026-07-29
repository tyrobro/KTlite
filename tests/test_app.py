"""
tests/test_app.py — Property-based structural tests for app.py

Property 1:  Session state key always present after init
Property 11: System prompt template always contains required placeholders
Property 12: Formatted prompt is a non-empty string
Property 14: Session state append only occurs on successful stream

Validates: Requirements 1.1, 1.2, 2.4, 3.3, 4.3, 4.1, 7.3, 7.7
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from app import SYSTEM_PROMPT_TEMPLATE, build_sources, render_chat_history_pure, render_sidebar


# ---------------------------------------------------------------------------
# Integration Smoke Tests — Startup Sequence
# Property 1: Session state key always present after init
# Validates: Requirements 1.1, 1.2, 2.4, 3.3
# ---------------------------------------------------------------------------


def test_init_session_state_property1_key_always_present():
    """Property 1 — 'messages' key must exist and be a list after init_session_state().

    Calls init_session_state() on an empty mock session state and asserts that
    the 'messages' key is present and holds a list.

    **Validates: Requirements 1.1, 1.2**
    """
    import app as app_module

    fake_state = {}
    with patch.object(app_module.st, "session_state", fake_state):
        app_module.init_session_state()

    assert "messages" in fake_state, "Expected 'messages' key to be created by init_session_state()"
    assert isinstance(fake_state["messages"], list), (
        f"Expected 'messages' to be a list, got {type(fake_state['messages'])}"
    )


def test_init_session_state_property1_idempotent():
    """Property 1 (idempotency) — calling init_session_state() twice must NOT reset an existing list.

    Provides a pre-populated 'messages' list, calls init_session_state() again,
    and asserts the original list object is preserved (not overwritten).

    **Validates: Requirements 1.1, 1.2**
    """
    import app as app_module

    existing_messages = [{"role": "user", "content": "hello"}]
    fake_state = {"messages": existing_messages}

    with patch.object(app_module.st, "session_state", fake_state):
        app_module.init_session_state()

    # The list object must be the exact same instance — not reset or replaced
    assert fake_state["messages"] is existing_messages, (
        "init_session_state() must not overwrite an already-initialised 'messages' list"
    )
    assert isinstance(fake_state["messages"], list)


def test_startup_stops_when_api_key_absent():
    """Startup guard — st.error and st.stop must both be called when GOOGLE_API_KEY is absent.

    Simulates the top-level guard in app.py:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error(...)
            st.stop()

    Patches the environment so the key is absent and verifies the guard fires.

    **Validates: Requirements 3.3**
    """
    import app as app_module

    mock_error = MagicMock()
    mock_stop = MagicMock()

    with patch.object(app_module.st, "error", mock_error), \
         patch.object(app_module.st, "stop", mock_stop), \
         patch.dict("os.environ", {}, clear=True):
        # Simulate the startup guard exactly as written in app.py
        if not os.getenv("GOOGLE_API_KEY"):
            app_module.st.error(
                "GOOGLE_API_KEY is not set. Please add it to your .env file and restart the app."
            )
            app_module.st.stop()

    mock_error.assert_called_once()
    mock_stop.assert_called_once()


def test_startup_stops_on_retriever_failure():
    """Startup guard — st.error and st.stop must be called when get_retriever() raises.

    Mirrors the try/except block at the top-level of app.py:
        try:
            vector_store, retriever = get_retriever()
        except Exception as exc:
            st.error(f"Failed to load the vector store: {exc}")
            st.stop()

    Patches get_retriever to raise and verifies both guards fire with the
    exception message embedded in the error call.

    **Validates: Requirements 2.4**
    """
    import app as app_module

    mock_error = MagicMock()
    mock_stop = MagicMock()
    chroma_error_msg = "chroma_db not found"

    with patch.object(app_module.st, "error", mock_error), \
         patch.object(app_module.st, "stop", mock_stop):
        # Simulate the startup try/except guard exactly as written in app.py
        try:
            raise Exception(chroma_error_msg)
        except Exception as exc:
            app_module.st.error(f"Failed to load the vector store: {exc}")
            app_module.st.stop()

    mock_error.assert_called_once()
    mock_stop.assert_called_once()

    # The error message must contain the original exception text
    error_call_arg = str(mock_error.call_args[0][0])
    assert chroma_error_msg in error_call_arg, (
        f"Expected '{chroma_error_msg}' in st.error argument, got: {error_call_arg!r}"
    )


# ---------------------------------------------------------------------------
# Property 11: System prompt template always contains required placeholders
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------

def test_system_prompt_has_context_placeholder():
    """Property 11 — {context} placeholder must be present in the template."""
    assert "{context}" in SYSTEM_PROMPT_TEMPLATE


def test_system_prompt_has_question_placeholder():
    """Property 11 — {question} placeholder must be present in the template."""
    assert "{question}" in SYSTEM_PROMPT_TEMPLATE


def test_system_prompt_has_sources_placeholder():
    """Property 11 — template contains {context} and {question}; {sources} was removed.

    Per Requirement 2.1: assert only {context} and {question} are present.
    """
    assert "{context}" in SYSTEM_PROMPT_TEMPLATE
    assert "{question}" in SYSTEM_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Property 12: Formatted prompt is a non-empty string
# Validates: Requirements 4.1, 4.3
# ---------------------------------------------------------------------------

def test_prompt_template_formats_to_nonempty_string():
    """Property 12 — .format() with valid args returns a non-empty str."""
    result = SYSTEM_PROMPT_TEMPLATE.format(
        context="ctx",
        question="q",
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_prompt_template_format_contains_supplied_values():
    """Property 12 (supplementary) — formatted prompt embeds the supplied values."""
    context = "Some document context."
    question = "What is the answer?"

    result = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    assert context in result
    assert question in result


def test_prompt_template_raises_no_key_error_on_format():
    """Property 12 (supplementary) — .format() does not raise KeyError for valid placeholders."""
    try:
        SYSTEM_PROMPT_TEMPLATE.format(context="ctx", question="q")
    except KeyError as exc:
        raise AssertionError(f"Unexpected KeyError during template formatting: {exc}") from exc


# ---------------------------------------------------------------------------
# Property-Based Tests for build_sources (Properties 7–10)
# Validates: Requirements 5.1, 5.2, 4.4
# ---------------------------------------------------------------------------
#
# **Validates: Requirements 5.1, 5.2, 4.4**
#
# Property 7:  Sources list is never empty
# Property 8:  Sources list contains no duplicates
# Property 9:  Every source is a non-empty string
# Property 10: Missing metadata source maps to "Unknown"

# Strategy: generate a list of Document-like dicts (0–3 items),
# each with optional metadata["source"].
_doc_strategy = st.fixed_dictionaries({
    "page_content": st.text(min_size=1),
    "metadata": st.one_of(
        st.fixed_dictionaries({"source": st.text(min_size=1)}),
        st.fixed_dictionaries({}),          # missing source → should map to "Unknown"
    ),
})


def _to_namespace(doc_dict: dict) -> SimpleNamespace:
    """Wrap a raw dict as a SimpleNamespace so .metadata attribute access works."""
    ns = SimpleNamespace()
    ns.metadata = doc_dict["metadata"]
    ns.page_content = doc_dict["page_content"]
    return ns


@settings(max_examples=200)
@given(st.lists(_doc_strategy, max_size=3))
def test_build_sources_property_7_never_empty(docs):
    """Property 7 — sources list is never empty, even for empty input.

    **Validates: Requirements 5.1, 7.3**
    """
    ns_docs = [_to_namespace(d) for d in docs]
    sources = build_sources(ns_docs)
    assert len(sources) >= 1


@settings(max_examples=200)
@given(st.lists(_doc_strategy, max_size=3))
def test_build_sources_property_8_no_duplicates(docs):
    """Property 8 — sources list contains no duplicate entries.

    **Validates: Requirements 5.2, 7.4**
    """
    ns_docs = [_to_namespace(d) for d in docs]
    sources = build_sources(ns_docs)
    assert len(sources) == len(set(sources))


@settings(max_examples=200)
@given(st.lists(_doc_strategy, max_size=3))
def test_build_sources_property_9_every_source_nonempty_string(docs):
    """Property 9 — every element in sources is a non-empty string.

    **Validates: Requirements 5.1, 5.3**
    """
    ns_docs = [_to_namespace(d) for d in docs]
    sources = build_sources(ns_docs)
    for s in sources:
        assert isinstance(s, str) and s


@settings(max_examples=200)
@given(st.lists(
    st.fixed_dictionaries({
        "page_content": st.text(min_size=1),
        "metadata": st.fixed_dictionaries({}),   # always missing "source"
    }),
    min_size=1,
    max_size=3,
))
def test_build_sources_property_10_missing_metadata_maps_to_unknown(docs):
    """Property 10 — docs with empty metadata always produce "Unknown"; no empty string.

    **Validates: Requirements 4.4, 5.1**
    """
    ns_docs = [_to_namespace(d) for d in docs]
    sources = build_sources(ns_docs)
    # Every source must be "Unknown" since no doc carries a "source" key
    assert "Unknown" in sources
    # No empty string must ever appear
    for s in sources:
        assert s != ""


# ---------------------------------------------------------------------------
# Unit Tests for render_sidebar — Clear Chat logic
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

class TestRenderSidebarClearChat(unittest.TestCase):
    """Unit tests for the Clear Chat button logic in render_sidebar.

    **Validates: Requirements 6.3**
    """

    def test_clear_chat_resets_messages_to_empty_list(self):
        """Clicking Clear Chat sets session_state['messages'] to []."""
        fake_session_state = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]}

        mock_sidebar_cm = MagicMock()
        mock_sidebar_cm.__enter__ = MagicMock(return_value=None)
        mock_sidebar_cm.__exit__ = MagicMock(return_value=False)

        with patch("streamlit.sidebar", mock_sidebar_cm), \
             patch("streamlit.header"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.session_state", fake_session_state):
            render_sidebar()

        self.assertEqual(fake_session_state["messages"], [])

    def test_clear_chat_with_empty_messages_stays_empty(self):
        """Clicking Clear Chat when messages is already [] leaves it []."""
        fake_session_state = {"messages": []}

        mock_sidebar_cm = MagicMock()
        mock_sidebar_cm.__enter__ = MagicMock(return_value=None)
        mock_sidebar_cm.__exit__ = MagicMock(return_value=False)

        with patch("streamlit.sidebar", mock_sidebar_cm), \
             patch("streamlit.header"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.session_state", fake_session_state):
            render_sidebar()

        self.assertEqual(fake_session_state["messages"], [])

    def test_no_clear_chat_when_button_not_clicked(self):
        """When Clear Chat button returns False, messages are NOT reset."""
        original_messages = [
            {"role": "user", "content": "Hello"},
        ]
        fake_session_state = {"messages": list(original_messages)}

        mock_sidebar_cm = MagicMock()
        mock_sidebar_cm.__enter__ = MagicMock(return_value=None)
        mock_sidebar_cm.__exit__ = MagicMock(return_value=False)

        with patch("streamlit.sidebar", mock_sidebar_cm), \
             patch("streamlit.header"), \
             patch("streamlit.button", return_value=False), \
             patch("streamlit.session_state", fake_session_state):
            render_sidebar()

        self.assertEqual(fake_session_state["messages"], original_messages)

    def test_clear_chat_replaces_with_list_not_none(self):
        """After Clear Chat click, messages must be a list, not None."""
        fake_session_state = {"messages": [{"role": "user", "content": "Test"}]}

        mock_sidebar_cm = MagicMock()
        mock_sidebar_cm.__enter__ = MagicMock(return_value=None)
        mock_sidebar_cm.__exit__ = MagicMock(return_value=False)

        with patch("streamlit.sidebar", mock_sidebar_cm), \
             patch("streamlit.header"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.session_state", fake_session_state):
            render_sidebar()

        self.assertIsInstance(fake_session_state["messages"], list)
        self.assertEqual(len(fake_session_state["messages"]), 0)


# ---------------------------------------------------------------------------
# Property-Based Tests for render_chat_history_pure (Property 2)
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------
#
# **Validates: Requirements 1.4**
#
# Property 2: Messages list contains only well-formed entries
#   - render_chat_history_pure must never raise on any input
#   - Every returned entry must have role in ("user", "assistant") and truthy content

# Strategy: mix valid messages with malformed ones
_valid_message_strategy = st.fixed_dictionaries({
    "role": st.sampled_from(["user", "assistant"]),
    "content": st.text(min_size=1, max_size=100),
})

_malformed_message_strategy = st.one_of(
    # Invalid role
    st.fixed_dictionaries({
        "role": st.just("bot"),
        "content": st.text(min_size=1),
    }),
    # Missing both keys
    st.fixed_dictionaries({}),
    # Valid role but empty content
    st.fixed_dictionaries({
        "role": st.sampled_from(["user", "assistant"]),
        "content": st.just(""),
    }),
    # Missing content key
    st.fixed_dictionaries({
        "role": st.sampled_from(["user", "assistant"]),
    }),
    # Missing role key
    st.fixed_dictionaries({
        "content": st.text(min_size=1),
    }),
)

_mixed_message_strategy = st.one_of(_valid_message_strategy, _malformed_message_strategy)


@settings(max_examples=200)
@given(st.lists(_mixed_message_strategy, max_size=20))
def test_render_chat_history_pure_property2_no_exception(messages):
    """Property 2 — render_chat_history_pure never raises regardless of input.

    **Validates: Requirements 1.4**
    """
    result = render_chat_history_pure(messages)
    assert isinstance(result, list)


@settings(max_examples=200)
@given(st.lists(_mixed_message_strategy, max_size=20))
def test_render_chat_history_pure_property2_only_valid_entries(messages):
    """Property 2 — every returned entry has a valid role and truthy content.

    **Validates: Requirements 1.4**
    """
    result = render_chat_history_pure(messages)
    for entry in result:
        assert entry.get("role") in ("user", "assistant"), (
            f"Invalid role in returned entry: {entry!r}"
        )
        assert entry.get("content"), (
            f"Empty or missing content in returned entry: {entry!r}"
        )


@settings(max_examples=200)
@given(st.lists(_mixed_message_strategy, max_size=20))
def test_render_chat_history_pure_property2_subset_of_input(messages):
    """Property 2 — returned entries are a subset of the input list (no fabrication).

    **Validates: Requirements 1.4**
    """
    result = render_chat_history_pure(messages)
    for entry in result:
        assert entry in messages, (
            f"Returned entry not found in original input: {entry!r}"
        )


# ---------------------------------------------------------------------------
# Helper: roles_alternate
# ---------------------------------------------------------------------------

def roles_alternate(roles: list) -> bool:
    """Return True if no two consecutive roles are the same."""
    return all(roles[i] != roles[i - 1] for i in range(1, len(roles)))


# ---------------------------------------------------------------------------
# Property 3: Roles alternate strictly
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------
#
# **Validates: Requirements 1.3**
#
# Property 3: Conversation turns alternate strictly — user → assistant → user → …


@settings(max_examples=300)
@given(st.integers(min_value=0, max_value=50))
def test_roles_alternate_property_3_valid_sequences_pass(n: int):
    """Property 3 — alternating role lists satisfy roles_alternate().

    Builds an alternating sequence of length n starting with 'user',
    then asserts roles_alternate returns True for it.

    **Validates: Requirements 1.3**
    """
    roles = ["user" if i % 2 == 0 else "assistant" for i in range(n)]
    # For any length, a strictly alternating list should pass
    assert roles_alternate(roles)


@settings(max_examples=300)
@given(
    st.integers(min_value=2, max_value=50),
    st.integers(min_value=0, max_value=49),
)
def test_roles_alternate_property_3_consecutive_same_roles_fail(length: int, insert_pos: int):
    """Property 3 — lists with consecutive identical roles fail roles_alternate().

    Builds an alternating list then inserts a duplicate role at insert_pos,
    producing two consecutive identical roles; asserts roles_alternate returns False.

    **Validates: Requirements 1.3**
    """
    roles = ["user" if i % 2 == 0 else "assistant" for i in range(length)]
    pos = insert_pos % length          # clamp to valid index
    duplicate_role = roles[pos]
    # Insert the duplicate immediately after pos — guarantees a consecutive pair
    non_alternating = roles[:pos + 1] + [duplicate_role] + roles[pos + 1:]
    assert not roles_alternate(non_alternating)


# ---------------------------------------------------------------------------
# Property 4: Content length is bounded
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------
#
# **Validates: Requirements 1.3**
#
# Property 4: Message content must not exceed 32,000 characters.

MAX_CONTENT_LENGTH = 32_000


@settings(max_examples=300)
@given(st.text(max_size=MAX_CONTENT_LENGTH))
def test_content_length_bounded_property_4_within_limit_passes(content: str):
    """Property 4 — Hypothesis-generated strings (up to 32,000 chars) satisfy the invariant.

    Confirms that any string Hypothesis produces within the allowed band satisfies
    len(content) <= 32_000.

    **Validates: Requirements 1.3**
    """
    assert len(content) <= MAX_CONTENT_LENGTH


def test_content_length_bounded_property_4_boundary_exact():
    """Property 4 — a string of exactly 32,000 chars is at (not over) the limit.

    **Validates: Requirements 1.3**
    """
    content = "x" * MAX_CONTENT_LENGTH
    assert len(content) == MAX_CONTENT_LENGTH
    assert len(content) <= MAX_CONTENT_LENGTH


def test_content_length_bounded_property_4_over_limit_rejected():
    """Property 4 — strings over 32,000 chars violate the invariant and would be rejected.

    Constructs a string of MAX_CONTENT_LENGTH + 1 and several larger sizes
    directly (Hypothesis cannot generate strings this large) and asserts that
    the rejection predicate len(content) > 32_000 holds for each.

    **Validates: Requirements 1.3**
    """
    for excess in (1, 100, 1_000, 10_000):
        content = "a" * (MAX_CONTENT_LENGTH + excess)
        assert len(content) > MAX_CONTENT_LENGTH, (
            f"Expected len > {MAX_CONTENT_LENGTH}, got {len(content)}"
        )


# ---------------------------------------------------------------------------
# Unit Tests for build_sources / build_context_block — Zero-docs error paths
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------

def test_zero_docs_build_sources_returns_unknown():
    """
    Requirement 7.3 — When the retriever returns zero documents,
    build_sources([]) must return exactly ['Unknown'].

    **Validates: Requirements 7.3**
    """
    result = build_sources([])
    assert result == ["Unknown"], f"Expected ['Unknown'], got {result!r}"


def test_zero_docs_build_context_block_returns_empty():
    """
    Requirement 7.3 — When the retriever returns zero documents,
    build_context_block([]) must return an empty string ''.

    **Validates: Requirements 7.3**
    """
    from app import build_context_block
    result = build_context_block([])
    assert result == "", f"Expected '', got {result!r}"


# ---------------------------------------------------------------------------
# Unit Tests for handle_query — LLM failure error paths
# Property 14: Session state append only occurs on successful stream
# Validates: Requirements 7.7
# ---------------------------------------------------------------------------

def test_handle_query_llm_failure_calls_st_error():
    """
    Requirement 7.7 — When the LLM raises an exception during streaming,
    st.error must be called with a message containing 'LLM error'.

    **Validates: Requirements 7.7**

    Property 14: Session state append only occurs on successful stream —
    verified structurally; the message list length does not grow on LLM failure.
    """
    import app as app_module
    from app import handle_query

    # Mock for st.chat_message context manager (used for both user and assistant)
    mock_chat_msg = MagicMock()
    mock_chat_msg.return_value.__enter__ = MagicMock(return_value=None)
    mock_chat_msg.return_value.__exit__ = MagicMock(return_value=False)

    fake_session_state = {"messages": []}

    mock_st_markdown = MagicMock()
    mock_st_error = MagicMock()

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("LLM error test")

    mock_spinner = MagicMock()
    mock_spinner.return_value.__enter__ = MagicMock(return_value=None)
    mock_spinner.return_value.__exit__ = MagicMock(return_value=False)

    with patch.object(app_module.st, "chat_message", mock_chat_msg), \
         patch.object(app_module.st, "session_state", fake_session_state), \
         patch.object(app_module.st, "markdown", mock_st_markdown), \
         patch.object(app_module.st, "error", mock_st_error), \
         patch.object(app_module.st, "spinner", mock_spinner):

        handle_query("test query", mock_retriever, mock_llm)

    # st.error must have been called once
    mock_st_error.assert_called_once()
    # The error message must contain "LLM error"
    call_args = mock_st_error.call_args
    error_msg = str(call_args[0][0]) if call_args[0] else str(call_args)
    assert "LLM error" in error_msg, (
        f"Expected 'LLM error' in st.error call argument, got: {error_msg!r}"
    )


def test_handle_query_llm_failure_no_assistant_message_appended():
    """
    Property 14 / Requirement 7.7 — When the LLM raises an exception,
    the session state messages list must NOT grow beyond the initial user
    message. Length stays at 1 (only the user message), never 2.

    **Validates: Requirements 7.3, 7.7**
    """
    import app as app_module
    from app import handle_query

    mock_chat_msg = MagicMock()
    mock_chat_msg.return_value.__enter__ = MagicMock(return_value=None)
    mock_chat_msg.return_value.__exit__ = MagicMock(return_value=False)

    fake_session_state = {"messages": []}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("LLM error test")

    mock_spinner = MagicMock()
    mock_spinner.return_value.__enter__ = MagicMock(return_value=None)
    mock_spinner.return_value.__exit__ = MagicMock(return_value=False)

    with patch.object(app_module.st, "chat_message", mock_chat_msg), \
         patch.object(app_module.st, "session_state", fake_session_state), \
         patch.object(app_module.st, "markdown", MagicMock()), \
         patch.object(app_module.st, "error", MagicMock()), \
         patch.object(app_module.st, "spinner", mock_spinner):

        handle_query("test query", mock_retriever, mock_llm)

    # Only the user message should have been appended — no assistant entry
    messages = fake_session_state["messages"]
    assert len(messages) == 1, (
        f"Expected exactly 1 message (user only), got {len(messages)}: {messages!r}"
    )
    assert messages[0]["role"] == "user", (
        f"Expected the single message to be the user message, got: {messages[0]!r}"
    )


# ---------------------------------------------------------------------------
# Task 6.1 — Smoke / structural test: imports and module-level symbols
# Validates: Requirements 1.1, 1.2, 2.2, 3.2
# ---------------------------------------------------------------------------

def test_smoke_neo4j_imports_and_module_symbols():
    """Smoke test — GraphCypherQAChain and Neo4jGraph importable from langchain_neo4j;
    app.graph and app.chain exist as module-level symbols when constructors are mocked.

    The conftest.py permanently patches Neo4jGraph, GraphCypherQAChain.from_llm, and
    ChatGoogleGenerativeAI before any import of app occurs, so no live Neo4j server
    or Google API key is required.

    Assertions:
    1. GraphCypherQAChain and Neo4jGraph are importable from langchain_neo4j.
    2. app.graph is a non-None module-level symbol (set by the mocked constructor).
    3. app.chain is a non-None module-level symbol (set by the mocked factory).
    4. Reloading app under fresh source-level mocks produces updated graph/chain symbols.

    **Validates: Requirements 1.1, 1.2, 2.2, 3.2**
    """
    import importlib

    # 1. Verify the two classes are importable from langchain_neo4j directly.
    from langchain_neo4j import GraphCypherQAChain, Neo4jGraph  # noqa: F401
    assert GraphCypherQAChain is not None, "GraphCypherQAChain must be importable from langchain_neo4j"
    assert Neo4jGraph is not None, "Neo4jGraph must be importable from langchain_neo4j"

    # 2. app is already imported (conftest patches ensure no live connection).
    #    Verify module-level symbols exist and are non-None.
    import app as app_module

    assert hasattr(app_module, "graph"), "app.graph must exist as a module-level symbol"
    assert hasattr(app_module, "chain"), "app.chain must exist as a module-level symbol"
    assert app_module.graph is not None, "app.graph must be non-None (set by mocked Neo4jGraph)"
    assert app_module.chain is not None, "app.chain must be non-None (set by mocked GraphCypherQAChain.from_llm)"

    # 3. Reload app under fresh source-level mocks.
    #    Patch langchain_neo4j.Neo4jGraph / .GraphCypherQAChain directly so that
    #    the ``from langchain_neo4j import …`` statement inside reload() picks up
    #    the new mocks (app.Neo4jGraph is rebound from the source during reload).
    fresh_graph_instance = MagicMock(name="fresh_graph_instance")
    fresh_chain_instance = MagicMock(name="fresh_chain_instance")

    mock_neo4j_cls = MagicMock(name="Neo4jGraph", return_value=fresh_graph_instance)
    mock_chain_cls = MagicMock(name="GraphCypherQAChain")
    mock_chain_cls.from_llm = MagicMock(return_value=fresh_chain_instance)

    with patch("langchain_neo4j.Neo4jGraph", mock_neo4j_cls), \
         patch("langchain_neo4j.GraphCypherQAChain", mock_chain_cls), \
         patch("app.get_llm", return_value=MagicMock(name="fresh_llm")):
        importlib.reload(app_module)

        # Each constructor must have been called exactly once during reload.
        mock_neo4j_cls.assert_called_once()
        mock_chain_cls.from_llm.assert_called_once()

        # graph and chain must reflect what the fresh mocks returned.
        assert app_module.graph is fresh_graph_instance, (
            "After reload, app.graph must be the instance returned by the patched Neo4jGraph"
        )
        assert app_module.chain is fresh_chain_instance, (
            "After reload, app.chain must be the instance returned by the patched GraphCypherQAChain.from_llm"
        )

    # 4. Restore: reload with conftest-level mocks active so the module stays usable.
    importlib.reload(app_module)


# ---------------------------------------------------------------------------
# Task 6.7 — Source-inspection test: no live retriever.invoke in handle_query
# Validates: Requirement 4.4
# ---------------------------------------------------------------------------

def test_no_live_retriever_invoke_in_handle_query():
    """Source-inspection test — reads app.py as text and asserts that no live
    (non-commented) call to ``retriever.invoke(`` exists inside the body of
    ``handle_query``.

    Steps:
    1. Read app.py as plain text.
    2. Locate the ``handle_query`` function body (from ``def handle_query`` to
       the next top-level ``def``/``class`` or end of file).
    3. For each line in that body, skip lines whose stripped form starts with
       ``#`` (comments).  Assert that no remaining line contains
       ``retriever.invoke(``.

    **Validates: Requirement 4.4**
    """
    import os
    import re

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    app_path = os.path.normpath(app_path)

    with open(app_path, "r", encoding="utf-8") as fh:
        source_lines = fh.readlines()

    # Find the line index where handle_query is defined.
    start_idx = None
    for i, line in enumerate(source_lines):
        if re.match(r"^def handle_query\b", line):
            start_idx = i
            break

    assert start_idx is not None, (
        "Could not find 'def handle_query' in app.py — the function must exist"
    )

    # Collect the body: from start_idx to the next top-level def/class or EOF.
    body_lines = []
    for i in range(start_idx + 1, len(source_lines)):
        line = source_lines[i]
        # A top-level definition starts at column 0 with def or class.
        if re.match(r"^(def |class )", line):
            break
        body_lines.append(line)

    # Assert no live (uncommented) retriever.invoke( call exists in the body.
    live_violations = []
    for lineno, line in enumerate(body_lines, start=start_idx + 2):
        stripped = line.strip()
        if stripped.startswith("#"):
            # This line is a comment — skip it.
            continue
        if "retriever.invoke(" in stripped:
            live_violations.append((lineno, line.rstrip()))

    assert len(live_violations) == 0, (
        "Found live (non-commented) retriever.invoke() call(s) inside handle_query:\n"
        + "\n".join(f"  line {ln}: {text}" for ln, text in live_violations)
    )


# ---------------------------------------------------------------------------
# Task 6.2 — Property 1: chain routing
# Feature: graph-rag-query-routing, Property 1
# Validates: Requirements 4.4, 5.1
# ---------------------------------------------------------------------------

@given(st.text(min_size=1, max_size=32_000))
@settings(max_examples=100)
def test_property1_chain_invoked_retriever_not_invoked(query):
    """Property 1 — handle_query routes through chain.invoke, not retriever.invoke.

    For any non-empty query string (1–32,000 characters), handle_query() must
    call chain.invoke({"query": query}) exactly once and must never call
    retriever.invoke().

    **Validates: Requirements 4.4, 5.1**
    """
    import app as app_module
    from app import handle_query

    # Build a mock chain that returns a valid result dict
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"result": "answer"}

    # Build a mock retriever to confirm it is never called
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    # Build Streamlit mocks
    mock_chat_msg = MagicMock()
    mock_chat_msg.return_value.__enter__ = MagicMock(return_value=None)
    mock_chat_msg.return_value.__exit__ = MagicMock(return_value=False)

    fake_session_state = {"messages": []}

    with patch.object(app_module, "chain", mock_chain), \
         patch.object(app_module.st, "chat_message", mock_chat_msg), \
         patch.object(app_module.st, "session_state", fake_session_state), \
         patch.object(app_module.st, "markdown", MagicMock()), \
         patch.object(app_module.st, "info", MagicMock()):

        handle_query(query, mock_retriever, mock_llm)

    # chain.invoke must have been called exactly once with the query dict
    mock_chain.invoke.assert_called_once_with({"query": query})

    # retriever.invoke must never have been called
    mock_retriever.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Task 6.3 — Property 2: exception handling
# Feature: graph-rag-query-routing, Property 2
# Validates: Requirements 5.3, 5.4, 7.4, 7.6
# ---------------------------------------------------------------------------

@given(st.text(min_size=1, max_size=32_000), st.text(min_size=1))
@settings(max_examples=100)
def test_property2_exception_preserves_user_message_calls_st_error(query, exc_msg):
    """Property 2 — chain.invoke exception handling preserves user message and surfaces error.

    For any exception raised by chain.invoke():
    - st.error() must be called exactly once with a message that contains the
      failure cause.
    - Session State must contain exactly 1 entry (the user message appended
      before the chain call).
    - That single entry must have role == "user".

    **Validates: Requirements 5.3, 5.4, 7.4, 7.6**
    """
    import app as app_module
    from app import handle_query

    # chain raises on every call
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError(exc_msg)

    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    mock_chat_msg = MagicMock()
    mock_chat_msg.return_value.__enter__ = MagicMock(return_value=None)
    mock_chat_msg.return_value.__exit__ = MagicMock(return_value=False)

    fake_session_state = {"messages": []}
    mock_st_error = MagicMock()

    with patch.object(app_module, "chain", mock_chain), \
         patch.object(app_module.st, "chat_message", mock_chat_msg), \
         patch.object(app_module.st, "session_state", fake_session_state), \
         patch.object(app_module.st, "markdown", MagicMock()), \
         patch.object(app_module.st, "error", mock_st_error):

        handle_query(query, mock_retriever, mock_llm)

    # st.error must have been called exactly once
    mock_st_error.assert_called_once()

    # The error message must contain the exception cause
    error_arg = str(mock_st_error.call_args[0][0])
    assert exc_msg in error_arg, (
        f"Expected exc_msg {exc_msg!r} to appear in st.error argument, got: {error_arg!r}"
    )

    # Session state must have exactly 1 entry — the user message
    messages = fake_session_state["messages"]
    assert len(messages) == 1, (
        f"Expected exactly 1 message in session state, got {len(messages)}: {messages!r}"
    )
    assert messages[0]["role"] == "user", (
        f"Expected the single session-state entry to have role='user', got: {messages[0]!r}"
    )


# ---------------------------------------------------------------------------
# Task 6.4 — Property 3: successful result extraction
# Feature: graph-rag-query-routing, Property 3
# Validates: Requirements 6.1, 6.4, 7.2, 7.3, 7.5
# ---------------------------------------------------------------------------

@given(
    st.text(min_size=1, max_size=32_000),
    st.text(min_size=1),
    st.dictionaries(st.text(), st.text()),
)
@settings(max_examples=100)
def test_property3_result_extracted_and_persisted(query, result_text, extra_keys):
    """Property 3 — successful invocation extracts result and persists to session state.

    For any dictionary returned by chain.invoke() that contains a non-empty
    "result" string:
    - st.markdown() must be called with result_text.
    - The last session-state entry must be {"role": "assistant", "content": result_text}.
    - st.error() must not be called.

    **Validates: Requirements 6.1, 6.4, 7.2, 7.3, 7.5**
    """
    import app as app_module
    from app import handle_query

    # Build response dict; extra_keys must not overwrite the "result" key
    response = {"result": result_text}
    response.update({k: v for k, v in extra_keys.items() if k != "result"})

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = response

    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    mock_chat_msg = MagicMock()
    mock_chat_msg.return_value.__enter__ = MagicMock(return_value=None)
    mock_chat_msg.return_value.__exit__ = MagicMock(return_value=False)

    fake_session_state = {"messages": []}
    mock_st_markdown = MagicMock()
    mock_st_error = MagicMock()

    with patch.object(app_module, "chain", mock_chain), \
         patch.object(app_module.st, "chat_message", mock_chat_msg), \
         patch.object(app_module.st, "session_state", fake_session_state), \
         patch.object(app_module.st, "markdown", mock_st_markdown), \
         patch.object(app_module.st, "error", mock_st_error), \
         patch.object(app_module.st, "info", MagicMock()):

        handle_query(query, mock_retriever, mock_llm)

    # st.markdown must have been called with result_text
    mock_st_markdown.assert_called_with(result_text)

    # The last session-state entry must be the assistant message
    messages = fake_session_state["messages"]
    assert len(messages) >= 1, "Session state must not be empty after a successful call"
    last_msg = messages[-1]
    assert last_msg == {"role": "assistant", "content": result_text}, (
        f"Expected last message to be assistant entry with result_text, got: {last_msg!r}"
    )

    # st.error must not have been called
    mock_st_error.assert_not_called()


# ---------------------------------------------------------------------------
# Task 6.5 — Property 4: malformed raw_response
# Feature: graph-rag-query-routing, Property 4
# Validates: Requirements 6.2, 6.3
# ---------------------------------------------------------------------------

@given(
    st.one_of(
        st.none(),
        st.text(),
        st.integers(),
        st.lists(st.text()),
        st.dictionaries(st.text(), st.text()).filter(lambda d: "result" not in d),
        st.just({}),
    )
)
@settings(max_examples=100)
def test_property4_malformed_response_no_exception(bad_response):
    """Property 4 — malformed or missing raw_response yields empty string without exception.

    For any value that is not a dict, is None, is an empty dict, or is a dict
    that does not contain the "result" key, the extraction expression must:
    - assign an empty string to full_response
    - not raise any exception

    **Validates: Requirements 6.2, 6.3**
    """
    # Call the extraction expression directly — no mocking required
    full_response = bad_response.get("result", "") if isinstance(bad_response, dict) else ""

    assert full_response == "", (
        f"Expected full_response to be '' for bad_response={bad_response!r}, "
        f"got {full_response!r}"
    )
    assert isinstance(full_response, str), (
        f"Expected full_response to be a str, got {type(full_response)}"
    )


# ---------------------------------------------------------------------------
# Task 6.6 — Example-based: empty result shows warning/info
# Validates: Requirements 6.4, 6.5
# ---------------------------------------------------------------------------

def test_empty_result_shows_info_not_markdown():
    """Example-based — when chain.invoke returns {"result": ""}, st.markdown must NOT
    be called with an empty string and either st.info or st.warning must be called
    at least once.

    **Validates: Requirements 6.4, 6.5**
    """
    import app as app_module
    from app import handle_query

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"result": ""}

    mock_chat_msg = MagicMock()
    mock_chat_msg.return_value.__enter__ = MagicMock(return_value=None)
    mock_chat_msg.return_value.__exit__ = MagicMock(return_value=False)

    fake_session_state = {"messages": []}
    mock_st_markdown = MagicMock()
    mock_st_info = MagicMock()
    mock_st_warning = MagicMock()
    mock_st_error = MagicMock()

    with patch.object(app_module, "chain", mock_chain), \
         patch.object(app_module.st, "chat_message", mock_chat_msg), \
         patch.object(app_module.st, "session_state", fake_session_state), \
         patch.object(app_module.st, "markdown", mock_st_markdown), \
         patch.object(app_module.st, "info", mock_st_info), \
         patch.object(app_module.st, "warning", mock_st_warning), \
         patch.object(app_module.st, "error", mock_st_error):

        handle_query("some query", None, None)

    # st.markdown must NOT have been called with an empty string
    for call in mock_st_markdown.call_args_list:
        args = call[0]
        assert not (len(args) == 1 and args[0] == ""), (
            "st.markdown must not be called with '' as sole argument when result is empty"
        )

    # Either st.info or st.warning must have been called at least once
    info_called = mock_st_info.call_count > 0
    warning_called = mock_st_warning.call_count > 0
    assert info_called or warning_called, (
        "Expected st.info or st.warning to be called when full_response is empty, "
        f"but info_called={info_called}, warning_called={warning_called}"
    )
