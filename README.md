# RAG-Project

A minimal end-to-end **GitContext** prototype for repository intelligence.

## Implemented capabilities

- Repository ingestion for source and docs (`.py`, `.md`, `.txt`)
- Intelligent code chunking for Python by **class/function/module** boundaries
- Chunk metadata including file path and last-modified timestamp
- Hybrid retrieval (**vector-style TF-IDF similarity + keyword/BM25-like scoring**)
- Lightweight reranking based on query-token overlap
- Staleness detection and automatic re-indexing for changed/removed files
- FastAPI endpoints for ingest/query
- CLI interface for local query flow

## Run CLI

```bash
python -m rag_project.cli /absolute/path/to/repo "Where is X defined?"
```

Note: for safety, the current prototype only ingests the active workspace root.

## Run API

```bash
pip install fastapi uvicorn
uvicorn rag_project.api:app --reload
```

Endpoints:
- `POST /ingest?repo_path=/absolute/path/to/repo`
- `POST /query` with JSON: `{"question":"...","top_k":5}`

## Tests

```bash
python -m pytest tests/test_rag_core.py
```
