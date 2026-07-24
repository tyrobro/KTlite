# Requirements Document

## Introduction

This document specifies the requirements for `ingest.py`, a modular Python ingestion module for a local, CPU-only Enterprise RAG (Retrieval-Augmented Generation) pipeline. The module loads `.txt` and `.pdf` documents from a local `./data` directory, splits them into overlapping text chunks while preserving source metadata, embeds the chunks using a local sentence transformer model, and persists the embedded vectors into a local ChromaDB vector store. The module must be importable as a library and executable as a standalone script.

## Glossary

- **Ingestion_Engine**: The `ingest.py` module responsible for end-to-end document loading, splitting, embedding, and storing.
- **Document_Loader**: The component within the Ingestion_Engine that reads `.txt` and `.pdf` files from `./data`.
- **Text_Splitter**: The component that divides raw document text into fixed-size overlapping chunks using `RecursiveCharacterTextSplitter`.
- **Embedding_Model**: The `HuggingFaceEmbeddings` wrapper around `all-MiniLM-L6-v2`, running locally on CPU.
- **Vector_Store**: The persistent ChromaDB instance stored at `./chroma_db`.
- **Chunk**: A text segment produced by the Text_Splitter, carrying source filename metadata.
- **Source_Metadata**: The `source` key in a Chunk's metadata dictionary, set to the originating filename (e.g., `os_part1.txt`).
- **ingest_documents()**: The public entry-point function exposed by the Ingestion_Engine.

---

## Requirements

### Requirement 1: Document Discovery and Loading

**User Story:** As a developer, I want the Ingestion_Engine to automatically discover and load all `.txt` and `.pdf` files from `./data`, so that I do not need to manually specify individual file paths.

#### Acceptance Criteria

1. WHEN `ingest_documents()` is called, THE Document_Loader SHALL scan the `./data` directory and collect all files with `.txt` or `.pdf` extensions.
2. WHEN a `.txt` file is discovered, THE Document_Loader SHALL load its full text content using `TextLoader` with UTF-8 encoding.
3. WHEN a `.pdf` file is discovered, THE Document_Loader SHALL load its full text content using `PyPDFLoader`, extracting text from all pages.
4. WHEN document loading completes and no files were successfully processed (whether due to an empty directory, permission errors, or all files failing), THE Ingestion_Engine SHALL log a warning message and return without raising an exception.
5. IF a file cannot be read due to a permission error or corrupted content, THEN THE Ingestion_Engine SHALL log an error message at `ERROR` level for that file, including the filename and exception message, and continue processing remaining files.
6. IF a permission error occurs while scanning the `./data` directory, THEN THE Ingestion_Engine SHALL log an error message and continue processing any files that were successfully discovered.

---

### Requirement 2: Document Chunking with Metadata Preservation

**User Story:** As a developer, I want each document split into manageable chunks with the source filename preserved in metadata, so that I can trace any retrieved chunk back to its origin file.

#### Acceptance Criteria

1. WHEN documents are loaded, THE Text_Splitter SHALL split each document into chunks using `RecursiveCharacterTextSplitter` with `chunk_size=400` and `chunk_overlap=40`.
2. THE Text_Splitter SHALL preserve the `source` key in each Chunk's metadata, set to the base filename (e.g., `"os_part1.txt"`) of the originating document.
3. WHEN a document produces zero chunks after splitting (e.g., the file is empty), THE Ingestion_Engine SHALL skip that document and log an informational message.
4. THE Text_Splitter SHALL produce chunks where the character length of each chunk does not exceed 400 characters, except when a single indivisible token exceeds 400 characters, in which case the oversized chunk SHALL be stored as-is regardless of its length.

---

### Requirement 3: Embedding Generation

**User Story:** As a developer, I want chunks embedded locally using a CPU-compatible sentence transformer, so that no external API calls or GPU are required.

#### Acceptance Criteria

1. THE Embedding_Model SHALL be initialized with `model_name="all-MiniLM-L6-v2"` using `HuggingFaceEmbeddings`.
2. THE Embedding_Model SHALL run inference on CPU only, without requiring a CUDA-capable device.
3. WHEN chunks are passed to the Embedding_Model, THE Embedding_Model SHALL produce one fixed-dimensional vector per Chunk; IF any chunk fails to generate an embedding, THE Ingestion_Engine SHALL fail the entire batch immediately and log the error.

---

### Requirement 4: Vector Store Persistence

**User Story:** As a developer, I want embedded chunks stored in a persistent local ChromaDB instance, so that the index survives process restarts without re-ingestion.

#### Acceptance Criteria

1. THE Vector_Store SHALL be initialized with `persist_directory="./chroma_db"` using `Chroma`.
2. WHEN `ingest_documents()` is called with a non-empty list of chunks, THE Vector_Store SHALL persist all embedded chunks to `./chroma_db` on disk.
3. WHEN `ingest_documents()` is called multiple times, THE Vector_Store SHALL add new chunks to the existing collection without deleting previously stored chunks.
4. IF the `./chroma_db` directory does not exist at startup, THEN THE Vector_Store SHALL create it automatically before writing data; IF directory creation fails due to insufficient permissions or disk space, THEN THE Ingestion_Engine SHALL fail immediately with a clear error message.

---

### Requirement 5: Public Interface and Script Entry Point

**User Story:** As a developer, I want to import `ingest_documents()` as a library function or run `ingest.py` directly from the command line, so that the module integrates flexibly into both automated pipelines and manual runs.

#### Acceptance Criteria

1. THE Ingestion_Engine SHALL expose an `ingest_documents()` function at module level that is importable without side effects (e.g., `from ingest import ingest_documents`).
2. WHEN `ingest.py` is executed directly (i.e., `__name__ == "__main__"`), THE Ingestion_Engine SHALL call `ingest_documents()` automatically.
3. WHEN `ingest_documents()` completes, THE Ingestion_Engine SHALL log the total number of chunks ingested, including when that count is zero.
4. THE `ingest_documents()` function SHALL accept an optional `data_dir` parameter (default: `"./data"`) and an optional `db_dir` parameter (default: `"./chroma_db"`), so that alternate paths can be supplied programmatically.

---

### Requirement 6: Logging and Observability

**User Story:** As a developer, I want structured log output during ingestion, so that I can monitor progress and diagnose failures without attaching a debugger.

#### Acceptance Criteria

1. THE Ingestion_Engine SHALL use Python's standard `logging` module for all output, configured at `INFO` level by default.
2. WHEN document loading begins, THE Ingestion_Engine SHALL log the total number of files discovered, including when that count is zero.
3. WHEN each file is loaded successfully, THE Ingestion_Engine SHALL log the filename and the number of chunks produced from it.
4. WHEN the Vector_Store write completes, THE Ingestion_Engine SHALL log a confirmation message including the total chunk count written.
5. IF an error occurs during loading or storing, THEN THE Ingestion_Engine SHALL log the error at `ERROR` level, including the filename and exception message.
