from __future__ import annotations

from dataclasses import dataclass

from .core import INDEX

try:
    from fastapi import FastAPI, HTTPException
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]


@dataclass
class QueryRequest:
    question: str
    top_k: int = 5


def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi is required to run the API. Install with `pip install fastapi uvicorn`.")

    app = FastAPI(title="GitContext API", version="0.1.0")

    @app.post("/ingest")
    def ingest(repo_path: str):
        try:
            return INDEX.ingest_repository(repo_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/query")
    def query(payload: QueryRequest):
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="question cannot be empty")
        return {"results": INDEX.query(payload.question, top_k=payload.top_k)}

    return app


app = create_app() if FastAPI is not None else None
