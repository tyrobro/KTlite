# Design Document: rag-ingestion-engine

## Overview

`ingest.py` is a single-file Python module implementing an end-to-end document ingestion pipeline for a local, CPU-only RAG system. It is designed to be imported as a library (`from ingest import ingest_documents`) or run directly as a script (`python ingest.py`). All processing is local — no external API calls or GPU required.

The module follows a linear pipeline: **Discover → Load → Split → Embed → Store**.

---

## Architecture

### Pipeline Flow

```
./data/
  ├── *.txt  ──► TextLoader
  └── *.pdf  ──► PyPDFLoader
                     │
                     ▼
          RecursiveCharacterTextSplitter
          (chunk_size=400, chunk_overlap=40)
          + source metadata normalization
                     │
                     ▼
          HuggingFaceEmbeddings
          (all-MiniLM-L6-v2, CPU)
                     │
                     ▼
          Chroma(persist_directory="./chroma_db")
```

---

## Components and Interfaces

### `_discover_files(data_dir: str) -> list[Path]`

Scans the given directory and returns a sorted list of `.txt` and `.pdf` file paths. Handles `PermissionError` during scan gracefully by logging at ERROR and returning whatever was collected.

**Input**: `data_dir` — string path to the directory containing source documents (default `"./data"`).  
**Output**: Sorted `list[pathlib.Path]` of discovered files (may be empty).

---

### `_load_documents(file_paths: list[Path]) -> list[Document]`

Loads each file using the appropriate LangChain loader. Normalises the `source` metadata field to the file's basename. Skips individual failing files with an ERROR log.

**Input**: List of `Path` objects from `_discover_files`.  
**Output**: `list[langchain_core.documents.Document]` — all successfully loaded documents with normalised metadata.

**Loader dispatch**:

| File suffix | Loader | Encoding |
|---|---|---|
| `.txt` | `TextLoader` | UTF-8 |
| `.pdf` | `PyPDFLoader` | N/A (binary) |

---

### `_split_documents(documents: list[Document]) -> list[Document]`

Splits loaded documents into overlapping chunks using `RecursiveCharacterTextSplitter`. LangChain propagates metadata (including `source`) to every chunk automatically.

**Input**: List of loaded `Document` objects.  
**Output**: Flat `list[Document]` of chunks, each with `page_content` and preserved metadata.

**Splitter configuration**: `chunk_size=400`, `chunk_overlap=40`.

---

### `ingest_documents(data_dir: str = "./data", db_dir: str = "./chroma_db") -> None`

Public orchestrator function. Chains all stages, logs progress, and handles terminal failures.

**Signature**: `def ingest_documents(data_dir: str = "./data", db_dir: str = "./chroma_db") -> None`

**Orchestration sequence**:
1. `_discover_files(data_dir)` → log file count
2. `_load_documents(file_paths)` → log or warn if empty → early return if nothing loaded
3. `_split_documents(documents)` → log chunk count
4. Initialise `HuggingFaceEmbeddings` and `Chroma` (see Error Handling)
5. `db.add_documents(chunks)` → log completion with chunk count

---

## Data Models

### `Document` (LangChain)

Reuses `langchain_core.documents.Document`. No custom data model is needed.

| Field | Type | Description |
|---|---|---|
| `page_content` | `str` | The text content of the document or chunk |
| `metadata` | `dict` | Arbitrary key/value pairs; `source` key holds the basename of the origin file |

### `source` metadata convention

The `source` field in `metadata` is normalised to `path.name` (basename only, e.g., `"os_part1.txt"`) during `_load_documents`. This is consistent with the existing `test_ingest.py` which reads `meta.get("source")`.

---

## Error Handling

| Scenario | Component | Behaviour |
|---|---|---|
| `PermissionError` scanning `data_dir` | `_discover_files` | Log ERROR; return partial file list |
| File unreadable / corrupt | `_load_documents` | Log ERROR with filename + message; skip file; continue |
| No documents loaded | `ingest_documents` | Log WARNING; return `None` — no write to ChromaDB |
| Empty document (zero chunks) | `_split_documents` | Log INFO; document is silently skipped from chunk list |
| `OSError` creating `db_dir` | `ingest_documents` | Raise `RuntimeError` with clear message; pipeline aborts |
| Embedding failure during `add_documents` | `ingest_documents` | Exception propagates to caller (fail-fast); entire batch discarded |

---

## Testing Strategy

Tests are located in `test_ingest.py` (already present in the project root). The strategy uses three levels:

**Property-based tests (Hypothesis)** — for logic that must hold across diverse inputs:
- Chunk size upper bound property
- Source metadata invariant across all chunks
- Embedding output count invariant

**Example tests** — for deterministic behaviour with known inputs:
- `.txt` and `.pdf` loading with sample files
- Additive persistence across two `ingest_documents()` calls
- Import side-effect free verification

**Edge-case tests** — for boundary and failure conditions:
- Empty `./data` directory
- All files failing to load
- Empty document producing zero chunks
- `OSError` on `db_dir` creation

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Single file vs package | Single `ingest.py` | Minimal footprint; no package boilerplate needed |
| Metadata normalisation | `path.name` (basename) | Consistent with how `test_ingest.py` reads `meta["source"]`; avoids full-path leakage |
| Per-file try/except | Yes | Partial failures should not abort the entire run |
| Embedding init location | Inside `ingest_documents()` | Avoids module-level side effects; model downloads only when needed |
| `add_documents` vs `from_documents` | `add_documents` on existing `Chroma` | Allows additive re-ingestion without wiping existing collection |
| Fail-fast on embedding error | Yes (let exception propagate) | Partial writes to ChromaDB could corrupt semantic search results |

---

## Correctness Properties

### Property 1: Chunk size upper bound
**Type**: PBT — Invariant  
**Validates: Requirements 2.1, 2.4**  
**Description**: For any text input, every chunk produced by `_split_documents` has `len(chunk.page_content) <= 400`, except when the chunk consists of a single indivisible token longer than 400 characters.

```
∀ chunk ∈ _split_documents(docs):
    len(chunk.page_content) <= 400
    OR chunk.page_content contains no whitespace split point
```

### Property 2: Source metadata invariant
**Type**: PBT — Invariant  
**Validates: Requirements 2.2**  
**Description**: For any document with `metadata["source"] = filename`, every chunk produced from that document must carry `metadata["source"] == filename`.

```
∀ doc ∈ documents, ∀ chunk ∈ split(doc):
    chunk.metadata["source"] == doc.metadata["source"]
```

### Property 3: Embedding output count invariant
**Type**: PBT — Invariant  
**Validates: Requirements 3.3**  
**Description**: The number of vectors produced by the Embedding_Model equals the number of input chunks.

```
∀ chunks of length N:
    len(embedding_model.embed_documents([c.page_content for c in chunks])) == N
```

### Property 4: Empty document produces no chunks
**Type**: Edge Case  
**Validates: Requirements 2.3**  
**Description**: A document whose `page_content` is `""` or only whitespace produces zero chunks from `_split_documents`.

### Property 5: Additive persistence
**Type**: Example  
**Validates: Requirements 4.3**  
**Description**: Calling `ingest_documents()` twice with different files results in a ChromaDB collection whose count equals the sum of chunks from both runs.

### Property 6: Import side-effect free
**Type**: Example  
**Validates: Requirements 5.1**  
**Description**: Importing `ingest` must not create any files, directories, or network connections. Verified by importing the module and asserting `./chroma_db` was not created.

### Property 7: No processable files — graceful return
**Type**: Edge Case  
**Validates: Requirements 1.4, 5.3**  
**Description**: When `./data` is empty or all files fail to load, `ingest_documents()` returns `None` without raising and without writing to ChromaDB.
