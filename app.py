"""
app.py — KTlite RAG Chat GUI
Phase 2: Streamlit-based conversational interface over the ChromaDB vector store.
"""

import os
from typing import TypedDict, Literal, List

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Message(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str   # non-empty, max 32 000 characters
    sources: list[str] # optional list of source filenames


class RetrievedChunk(TypedDict):
    page_content: str           # chunk text; never empty for a valid document
    metadata: dict              # must contain "source": str; falls back to "Unknown"


# ---------------------------------------------------------------------------
# System Prompt Template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the user's question using ONLY the \
information provided in the context below. Do not infer, guess, or use \
knowledge outside of this context.

If the context does not contain enough information to answer the question, \
respond with exactly:
"I do not have enough information to answer that based on the provided documents."

Context:
{context}

Question:
{question}
"""


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Idempotent — safe to call on every rerun."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


# ---------------------------------------------------------------------------
# Pure Helper Functions (no Streamlit imports/references)
# ---------------------------------------------------------------------------

def build_context_block(docs: list) -> str:
    """Join page_content of each doc with '\\n\\n'. Returns '' for empty input."""
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)


def build_sources(docs: list) -> list:
    """
    Extract and deduplicate source filenames from doc metadata.
    Preserves insertion order via dict.fromkeys.
    Returns ['Unknown'] for empty input.
    """
    if not docs:
        return ["Unknown"]
    return list(dict.fromkeys(
        doc.metadata.get("source", "Unknown") for doc in docs
    ))


# ---------------------------------------------------------------------------
# Cached Resource Loading
# ---------------------------------------------------------------------------

@st.cache_resource
def get_retriever():
    """
    Load the ChromaDB vector store and wrap it as a LangChain retriever.
    Decorated with @st.cache_resource so the heavy embedding model and Chroma
    client are created only once per server process and reused on every rerun.

    Returns:
        tuple[Chroma, VectorStoreRetriever]: the vector store and its retriever.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return vector_store, retriever


@st.cache_resource
def get_llm():
    """
    Initialise the Gemini LLM.
    Decorated with @st.cache_resource so the client is created only once per
    server process and reused on every rerun.

    Returns:
        ChatGoogleGenerativeAI: the LLM instance ready for .stream() calls.
    """
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """
    Render the application sidebar.

    Displays the app title and a Clear Chat button. Clicking the button resets
    session state messages to an empty list, triggering an immediate rerun
    that clears the chat window.
    """
    with st.sidebar:
        st.header("📚 KTlite — RAG Chat")
        if st.button("Clear Chat"):
            st.session_state["messages"] = []


def render_chat_history_pure(messages: list) -> list[dict]:
    """
    Pure variant of render_chat_history — no Streamlit calls.

    Filters ``messages`` to only the entries that would be rendered:
    - ``role`` must be exactly ``"user"`` or ``"assistant"``
    - ``content`` must be truthy (non-empty string)

    Returns the list of valid entries in their original order.
    Used by property-based tests (Requirement 1.4).
    """
    valid: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            valid.append(msg)
    return valid


def render_chat_history() -> None:
    """
    Render the full chat history stored in ``st.session_state["messages"]``.

    Iterates the message list and displays each valid entry inside a
    ``st.chat_message`` bubble using ``st.markdown``.  Entries whose
    ``role`` is not ``"user"`` or ``"assistant"``, or whose ``content``
    is falsy, are silently skipped. Extracted sources are rendered in an expander.

    Requirement: 1.4
    """
    for msg in st.session_state["messages"]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            with st.chat_message(role):
                st.markdown(content)
                
                # Render sources cleanly if present
                sources = msg.get("sources")
                if sources:
                    with st.expander("📚 **Retrieved Source Documents**"):
                        for source in sources:
                            st.caption(f"📄 `{source}`")


# ---------------------------------------------------------------------------
# Query Pipeline
# ---------------------------------------------------------------------------

def handle_query(query: str, retriever, llm) -> None:
    """
    Execute the full RAG query pipeline for a single user submission.

    Steps:
        1. Render and persist the user message bubble.
        2. Retrieve up to 3 relevant chunks from the vector store.
        3. Build the Context Block and deduplicate source filenames.
        4. Format the System Prompt with context and question.
        5-6. Stream the LLM response into an assistant chat bubble.
        7. Persist the full assistant response to Session State.
    """
    # Step 1 — Render and persist user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["messages"].append({"role": "user", "content": query})

    # Step 2 — Retrieve chunks
    docs = retriever.invoke(query)          # returns List[Document], len ∈ {0..3}

    # Step 3 — Build Context Block and deduplicate sources
    context = build_context_block(docs)
    sources = build_sources(docs)

    # Step 4 — Format System Prompt
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        question=query,
    )

    # Step 5-6 — Stream LLM response
    with st.chat_message("assistant"):
        try:
            # Wait for the full response instead of streaming to avoid chunk parsing errors
            with st.spinner("Thinking..."):
                response = llm.invoke(prompt)
            
            # Safely extract the text content
            if isinstance(response.content, str):
                full_response = response.content
            elif isinstance(response.content, list):
                # Handle cases where LangChain returns a list of content blocks
                full_response = "".join(block.get("text", "") for block in response.content if isinstance(block, dict))
            else:
                full_response = str(response.content)

            # Render the clean text to the UI
            st.markdown(full_response)
            
            # Display the retrieved files directly beneath the answer
            if sources and sources != ["Unknown"]:
                with st.expander("📚 **Retrieved Source Documents**"):
                    for source in sources:
                        st.caption(f"📄 `{source}`")

        except Exception as exc:
            st.error(f"LLM error: {exc}")
            return   # do NOT append to session state

    # Step 7 — Persist assistant response
    st.session_state["messages"].append(
        {
            "role": "assistant", 
            "content": full_response,
            "sources": sources if sources != ["Unknown"] else []
        }
    )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Streamlit application entry point.

    All top-level Streamlit calls are deferred inside this function so that
    ``import app`` (or ``from app import …``) does not spin up the server or
    trigger Streamlit runtime side-effects during testing.
    """
    # Step 1: Load environment variables
    load_dotenv()

    # Step 2: Guard — check GOOGLE_API_KEY before anything else
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is not set. Please add it to your .env file and restart the app."
        )
        st.stop()

    # Step 3: Load vector store and retriever (cached)
    try:
        vector_store, retriever = get_retriever()
    except Exception as exc:
        st.error(f"Failed to load the vector store: {exc}")
        st.stop()

    # Step 4: Load LLM (cached)
    llm = get_llm()

    # Step 5: Initialise session state
    init_session_state()

    # Step 6: Render sidebar
    render_sidebar()

    # Step 7: Render chat history
    render_chat_history()

    # Step 8: Chat input and query handling
    if query := st.chat_input("Ask a question about your documents..."):
        handle_query(query, retriever, llm)


# Guard: run main() only when Streamlit executes this file as the active
# script — not when the module is imported during testing or tooling.
# ``st.runtime.exists()`` returns True only inside a live Streamlit session.
if st.runtime.exists():
    main()