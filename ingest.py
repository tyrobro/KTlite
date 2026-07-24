"""
ingest.py — Document ingestion pipeline for local, CPU-only RAG system.

Pipeline: Discover → Load → Split → Embed → Store

Usage:
    As a library : from ingest import ingest_documents
    As a script  : python ingest.py
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline functions (stubs — implemented in subsequent tasks)
# ---------------------------------------------------------------------------

def _discover_files(data_dir: str) -> list:
    """Scan data_dir and return a sorted list of .txt and .pdf Paths."""
    path = Path(data_dir)
    collected: list[Path] = []
    try:
        for entry in path.glob("*"):
            if entry.suffix.lower() in (".txt", ".pdf"):
                collected.append(entry)
    except PermissionError as e:
        logger.error("Permission denied scanning directory '%s': %s", data_dir, e)
        return sorted(collected)
    return sorted(collected)


def _load_documents(file_paths: list) -> list:
    """Load each file with the appropriate LangChain loader.

    Dispatches .txt files to TextLoader (UTF-8) and .pdf files to PyPDFLoader.
    Normalises doc.metadata["source"] to the file's basename for every loaded
    Document. Skips any file that raises an exception, logging at ERROR level.

    Args:
        file_paths: List of Path objects to load.

    Returns:
        Combined list[Document] from all successfully loaded files.
    """
    all_docs = []
    for path in file_paths:
        try:
            suffix = path.suffix.lower()
            if suffix == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
            elif suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                logger.error("Unsupported file type '%s': skipping", path.name)
                continue

            docs = loader.load()

            # Normalise source metadata to basename only
            for doc in docs:
                doc.metadata["source"] = path.name

            all_docs.extend(docs)

        except Exception as e:
            logger.error("Failed to load '%s': %s", path.name, e)
            continue

    return all_docs


def _split_documents(documents: list) -> list:
    """Split documents into overlapping chunks.

    Creates a RecursiveCharacterTextSplitter with chunk_size=400 and
    chunk_overlap=40. LangChain propagates metadata (including 'source')
    to every chunk automatically. Logs INFO for any source file that
    produces zero chunks.

    Args:
        documents: List of loaded Document objects.

    Returns:
        Flat list[Document] of chunks with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(documents)

    # Detect source files that contributed zero chunks
    sources_in_chunks = {chunk.metadata.get("source") for chunk in chunks}
    sources_in_docs = {doc.metadata.get("source") for doc in documents}
    for source in sources_in_docs - sources_in_chunks:
        logger.info("No chunks produced from '%s'", source)

    return chunks


def ingest_documents(data_dir: str = "./data", db_dir: str = "./chroma_db") -> None:
    """
    Public entry point.  Orchestrates the full ingestion pipeline.

    Args:
        data_dir: Path to the directory containing source documents.
        db_dir:   Path to the ChromaDB persist directory.
    """
    # Step 1 — Discover
    file_paths = _discover_files(data_dir)
    logger.info("Discovered %d file(s) in %s", len(file_paths), data_dir)

    # Step 2 — Load
    documents = _load_documents(file_paths)
    if not documents:
        logger.warning("No documents successfully loaded.")
        return

    # Step 3 — Split
    chunks = _split_documents(documents)
    logger.info("Split into %d chunk(s) total.", len(chunks))

    # Step 4 — Embed + Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    except OSError as e:
        raise RuntimeError(f"Cannot create vector store at {db_dir}: {e}") from e

    # Step 5 — Write
    db.add_documents(chunks)
    logger.info("Ingestion complete. %d chunk(s) written to %s.", len(chunks), db_dir)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingest_documents()
