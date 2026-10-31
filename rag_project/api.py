"""FastAPI Backend Server for GitContext / RAG-Project."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .config import settings
from .core import INDEX
from .evaluator import CodebaseEvaluator

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Developer query, error message, or symbol name")
    top_k: int = Field(default=5, ge=1, le=50)


class IngestRequest(BaseModel):
    repo_path: Optional[str] = Field(default=None, description="Path to repository. Defaults to current directory.")


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install with `pip install fastapi uvicorn`.")

    app = FastAPI(
        title="GitContext API",
        version=settings.VERSION,
        description="End-to-End Codebase Intelligence & Developer Copilot with Hybrid Retrieval & Reranking",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index():
        """Serves the interactive web UI dashboard."""
        if frontend_path.exists():
            return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>GitContext API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>"
        )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.VERSION,
            "indexed_chunks": len(INDEX.chunks),
            "tracked_files": len(INDEX.file_state),
        }

    @app.post("/ingest")
    def ingest(payload: Optional[IngestRequest] = None, repo_path: Optional[str] = None):
        target_path = (payload.repo_path if payload and payload.repo_path else None) or repo_path
        try:
            res = INDEX.ingest_repository(target_path)
            return {
                "status": "success",
                "chunks_indexed": len(INDEX.chunks),
                "summary": res,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/query")
    def query(payload: QueryRequest):
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="question cannot be empty")
        results = INDEX.query(payload.question, top_k=payload.top_k)
        return {"results": results, "total_found": len(results)}

    @app.post("/ask")
    def ask(payload: QueryRequest):
        """Generates grounded answer with exact file, line, and author citations."""
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="question cannot be empty")
        return INDEX.ask(payload.question, top_k=payload.top_k)

    @app.post("/sync")
    def sync():
        """Incrementally syncs modified/stale files."""
        return INDEX.sync()

    @app.get("/evaluate")
    def evaluate(top_k: int = Query(default=5, ge=1, le=20)):
        """Runs evaluation benchmark measuring Recall@K, Precision, Citation accuracy, and Latency."""
        evaluator = CodebaseEvaluator(INDEX)
        return evaluator.run_benchmark(top_k=top_k)

    return app


app = create_app() if FastAPI is not None else None
