"""
ingest.py — Document ingestion pipeline for local, CPU-only RAG system.

Pipeline: Discover → Load → Split → Embed → Store

Usage:
    As a library : from ingest import ingest_documents
    As a script  : python ingest.py
"""
from dotenv import load_dotenv
import logging
import os
import time
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graph_transformers.llm import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
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
                loader = PDFPlumberLoader(str(path))
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


def _extract_graph_documents(
    documents: list,
    llm_transformer: LLMGraphTransformer,
    graph: Neo4jGraph,
) -> list:
    """Run throttled per-document graph extraction and write results to graph store.

    Returns the accumulated list of GraphDocument objects written to the store.
    """
    graph_documents: list = []
    for doc in documents:
        try:
            result = llm_transformer.convert_to_graph_documents([doc])
            if result:
                graph_documents.extend(result)
            else:
                logger.warning(
                    "No graph documents extracted from '%s'",
                    doc.metadata.get("source", "unknown"),
                )
        except Exception as exc:
            logger.error(
                "Graph extraction failed for '%s': %s",
                doc.metadata.get("source", "unknown"),
                exc,
            )
        finally:
            time.sleep(4)

    graph.add_graph_documents(
        graph_documents, baseEntityLabel=True, include_source=True
    )
    logger.info(
        "Knowledge graph updated: %d graph document(s) stored.", len(graph_documents)
    )
    return graph_documents


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

    # Initialise Neo4j connection and LLM graph transformer
    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)

    # Step 2.5 — Extract knowledge graph from full (un-split) documents
    logger.info("Extracting knowledge graph from %d document(s)...", len(documents))
    _extract_graph_documents(documents, llm_transformer, graph)

    # Step 3 — Split
    chunks = _split_documents(documents)
    logger.info("Split into %d chunk(s) total.", len(chunks))

    # Step 4 — Embed + Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    except OSError as e:
        raise RuntimeError(f"Cannot create vector store at {db_dir}: {e}") from e

    # Initialise record manager for deduplication / sync (idempotent)
    record_manager = SQLRecordManager(
        namespace="chroma/rag_docs",
        db_url="sqlite:///record_manager_cache.sql",
    )
    record_manager.create_schema()

    # Step 5 — Write
    index_result = index(chunks, record_manager, db, cleanup="full", source_id_key="source")
    print(index_result)
    logger.info("Ingestion complete: %s", index_result)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingest_documents()
