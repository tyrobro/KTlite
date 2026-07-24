# Implementation Plan

## Overview

Six sequential tasks build `ingest.py` from scaffold to fully verified end-to-end pipeline. Each task is self-contained and verifiable before moving to the next.

## Tasks

- [x] 1. Set up module scaffold and logging
  - Create `ingest.py` at the project root with all required imports: `pathlib`, `logging`, `langchain_community.document_loaders` (`TextLoader`, `PyPDFLoader`), `langchain.text_splitter` (`RecursiveCharacterTextSplitter`), `langchain_huggingface` (`HuggingFaceEmbeddings`), `langchain_chroma` (`Chroma`)
  - Configure module-level logger using `logging.basicConfig` at `INFO` level with timestamp format
  - Define `logger = logging.getLogger(__name__)`
  - Add the `if __name__ == "__main__"` guard stub (calls `ingest_documents()`)
  - **Verification**: Import the module from a Python REPL — no exceptions, no files created, no model downloads triggered
  - **References**: Requirements §5.1 (importable without side effects), §6.1 (logging at INFO)

- [x] 2. Implement `_discover_files(data_dir)`
  - Accept `data_dir: str` and convert to `pathlib.Path`
  - Use `Path.glob("*")` to iterate entries; filter for `.txt` and `.pdf` suffixes (case-insensitive via `.lower()`)
  - Return a sorted `list[Path]` for deterministic ordering
  - Wrap the scan in `try/except PermissionError`: log at ERROR and return whatever was collected so far
  - **Verification**: Call with `"./data"` — returns paths for `course_structure.txt` and `os_part1.txt`
  - **References**: Requirements §1.1, §1.6

- [x] 3. Implement `_load_documents(file_paths)`
  - Accept `file_paths: list[Path]`
  - For each path: dispatch `.txt` to `TextLoader(str(path), encoding="utf-8").load()`, `.pdf` to `PyPDFLoader(str(path)).load()`
  - After loading, normalise `doc.metadata["source"]` to `path.name` (basename) for every `Document` in the result list
  - Wrap each individual file load in `try/except Exception`: log at ERROR with filename and exception message; continue to next file
  - Return combined `list[Document]` from all successfully loaded files
  - **Verification**: Load `./data/course_structure.txt` — document text contains "CS301"; `metadata["source"]` equals `"course_structure.txt"` (not full path)
  - **References**: Requirements §1.2, §1.3, §1.5, §2.2

- [x] 4. Implement `_split_documents(documents)`
  - Accept `documents: list[Document]`
  - Create `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)`
  - Call `splitter.split_documents(documents)` to produce chunks (LangChain propagates metadata automatically)
  - For each source file in `documents` that contributes zero chunks to the result, log an INFO message naming the file
  - Return the flat `list[Document]` of chunks
  - **Verification**: Split the two sample `.txt` files — every chunk has `len(page_content) <= 400`; every chunk carries the correct `source` basename in metadata
  - **References**: Requirements §2.1, §2.2, §2.3, §2.4

- [x] 5. Implement `ingest_documents(data_dir, db_dir)` orchestrator
  - Signature: `def ingest_documents(data_dir: str = "./data", db_dir: str = "./chroma_db") -> None`
  - Step 1 — discover: `file_paths = _discover_files(data_dir)`; log `"Discovered {len(file_paths)} file(s) in {data_dir}"`
  - Step 2 — load: `documents = _load_documents(file_paths)`; if empty, log WARNING `"No documents successfully loaded."` and return
  - Step 3 — split: `chunks = _split_documents(documents)`; log `"Split into {len(chunks)} chunk(s) total."`
  - Step 4 — embed + store: initialise `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`; wrap `Chroma(persist_directory=db_dir, embedding_function=embeddings)` in `try/except OSError` — on failure, raise `RuntimeError(f"Cannot create vector store at {db_dir}: {e}")`
  - Step 5 — write: `db.add_documents(chunks)`; log `"Ingestion complete. {len(chunks)} chunk(s) written to {db_dir}."`
  - **Verification**: Run `python ingest.py` from project root — `./chroma_db` is created; running `test_ingest.py` shows `Total chunks stored in DB` > 0 and `Source File` shows the correct basename
  - **References**: Requirements §3.1, §3.2, §3.3, §4.1, §4.2, §4.3, §4.4, §5.2, §5.3, §5.4, §6.2, §6.3, §6.4, §6.5

- [x] 6. Verify end-to-end correctness against existing test harness
  - Run `python ingest.py` from `d:\Ache Kaam\Projects\KTlite\`
  - Run `python test_ingest.py` and confirm:
    - `Total chunks stored in DB` is a positive integer
    - `Source File` for each printed chunk shows only the filename (e.g., `"course_structure.txt"`), not a full path
    - `Text Snippet` contains recognisable content from the sample files
  - Run `python ingest.py` a second time and re-run `test_ingest.py` — chunk count must be greater than or equal to the first run (additive behaviour)
  - **References**: Requirements §4.3, Design Property 5 (additive persistence)

## Task Dependency Graph

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"] },
    { "wave": 2, "tasks": ["2"] },
    { "wave": 3, "tasks": ["3"] },
    { "wave": 4, "tasks": ["4"] },
    { "wave": 5, "tasks": ["5"] },
    { "wave": 6, "tasks": ["6"] }
  ]
}
```

All tasks are strictly sequential. Each task builds on the functions defined in the previous task.

## Notes

- Do NOT generate `requirements.txt`, test files, or environment setup steps — these are out of scope.
- The `test_ingest.py` file in the project root is the existing verification harness; task 6 uses it as-is.
- All paths (`./data`, `./chroma_db`) are relative to the project root (`d:\Ache Kaam\Projects\KTlite\`).
- The `all-MiniLM-L6-v2` model is downloaded on first use by `sentence-transformers`; subsequent runs use the local cache.
