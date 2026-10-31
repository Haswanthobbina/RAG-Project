"""Core Repository Intelligence Engine with AST Chunking, Git Mining, and Hybrid Retrieval."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rag_project.bm25_index import BM25SearchEngine
from rag_project.config import settings
from rag_project.generator import CodeCopilotGenerator
from rag_project.git_miner import GitMiner
from rag_project.retriever import HybridRetriever
from rag_project.vector_store import VectorStore


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


class RepositoryIndex:
    """Production Repository Intelligence Index."""

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.file_state: Dict[str, Dict[str, str]] = {}
        self.git_miner: Optional[GitMiner] = None
        self.vector_store = VectorStore()
        self.bm25_engine = BM25SearchEngine()
        self.retriever = HybridRetriever(self.vector_store, self.bm25_engine)
        self.generator = CodeCopilotGenerator()

    @staticmethod
    def _chunk_python(path: Path, content: str) -> Iterable[Tuple[str, str, int, int, str]]:
        """Parses Python code using AST to identify functions, classes, and module docstrings."""
        lines = content.splitlines()
        try:
            parsed = ast.parse(content)
        except SyntaxError:
            # Fallback to module chunk if syntax error
            yield "module", path.stem, 1, max(1, len(lines)), content
            return

        has_structural_nodes = False
        for node in parsed.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                has_structural_nodes = True
                start = node.lineno
                end = getattr(node, "end_lineno", node.lineno)
                chunk_text = "\n".join(lines[start - 1 : end])
                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
                chunk_name = node.name
                yield chunk_type, chunk_name, start, end, chunk_text

        # Capture top-level module header or docstring
        if parsed.body:
            first_lineno = parsed.body[0].lineno
            if first_lineno > 1:
                header = "\n".join(lines[: first_lineno - 1]).strip()
                if header:
                    yield "module", path.stem, 1, first_lineno - 1, header

        if not has_structural_nodes and content.strip():
            yield "module", path.stem, 1, max(1, len(lines)), content

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_chunk(
        self,
        path: Path,
        chunk_type: str,
        name: str,
        start: int,
        end: int,
        text: str,
        git_meta: Optional[Dict[str, str]] = None,
    ) -> Chunk:
        posix_path = path.as_posix()
        chunk_id = f"{posix_path}::{chunk_type}::{name}::{start}-{end}"
        metadata = {
            "file_path": str(path),
            "file_path_posix": posix_path,
            "chunk_type": chunk_type,
            "name": name,
            "start_line": str(start),
            "end_line": str(end),
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        if git_meta:
            metadata.update(git_meta)
        return Chunk(chunk_id=chunk_id, text=text, metadata=metadata)

    def ingest_repository(
        self,
        repo_path: Optional[str] = None,
        include_extensions: Tuple[str, ...] = settings.SUPPORTED_EXTENSIONS,
    ) -> Dict[str, List[str]]:
        """Ingests repository files, extracts AST chunks, attaches Git metadata, and builds indices."""
        target_dir = Path(repo_path).resolve() if repo_path else Path.cwd().resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"repo_path directory does not exist: {target_dir}")

        self.git_miner = GitMiner(target_dir)

        added: List[str] = []
        updated: List[str] = []
        removed: List[str] = []

        # Discover all eligible files
        current_files: Dict[str, Path] = {}
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in settings.IGNORED_DIRS and not d.startswith(".")]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in include_extensions:
                    current_files[str(file_path)] = file_path

        # Handle removed files
        for tracked_file in list(self.file_state.keys()):
            if tracked_file not in current_files:
                removed.append(tracked_file)
                self.file_state.pop(tracked_file, None)
                self.chunks = [c for c in self.chunks if c.metadata["file_path"] != tracked_file]

        # Ingest and detect modified files
        for file_key, absolute_path in current_files.items():
            try:
                text = absolute_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            digest = self._hash_text(text)
            mtime = str(absolute_path.stat().st_mtime_ns)
            state = self.file_state.get(file_key)

            if state and state["digest"] == digest and state["mtime"] == mtime:
                continue

            if state:
                updated.append(file_key)
            else:
                added.append(file_key)

            # Clear old chunks for this file
            self.chunks = [c for c in self.chunks if c.metadata["file_path"] != file_key]

            # Get Git history & author attribution
            git_meta = self.git_miner.get_file_metadata(absolute_path)

            if absolute_path.suffix.lower() == ".py":
                for chunk_type, name, start, end, chunk_text in self._chunk_python(absolute_path, text):
                    self.chunks.append(
                        self._build_chunk(absolute_path, chunk_type, name, start, end, chunk_text, git_meta)
                    )
            else:
                chunk_type = "document" if absolute_path.suffix.lower() in (".md", ".txt") else "code"
                lines = text.splitlines()
                self.chunks.append(
                    self._build_chunk(
                        absolute_path,
                        chunk_type,
                        absolute_path.stem,
                        1,
                        max(1, len(lines)),
                        text,
                        git_meta,
                    )
                )

            self.file_state[file_key] = {"digest": digest, "mtime": mtime}

        # Build dual indices
        self._recompute_indices()

        if self.git_miner:
            self.git_miner.update_last_commit()

        return {"added": added, "updated": updated, "removed": removed}

    def _recompute_indices(self) -> None:
        """Re-indexes chunks into both Dense Vector Store and BM25 Search Engine."""
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "file_path": c.metadata.get("file_path", ""),
                "name": c.metadata.get("name", ""),
                "start_line": int(c.metadata.get("start_line", 1)),
                "end_line": int(c.metadata.get("end_line", 1)),
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        self.vector_store.build_index(chunk_dicts)
        self.bm25_engine.build_index(chunk_dicts)

    def query(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the repository using Hybrid Retrieval (Vector + BM25) and Reranking."""
        return self.retriever.retrieve(prompt, top_k=top_k)

    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Generates copilot answer with grounded citations."""
        contexts = self.query(question, top_k=top_k)
        return self.generator.generate(question, contexts)

    def sync(self) -> Dict[str, Any]:
        """Performs incremental synchronization of changed files."""
        if not self.git_miner or not self.git_miner.is_git_repo:
            # Fallback to filesystem mtime scan
            target_dir = Path.cwd().resolve()
            return self.ingest_repository(str(target_dir))

        changed_files = self.git_miner.detect_changed_files()
        ingest_res = self.ingest_repository(str(self.git_miner.repo_path))
        return {
            "git_changed_files": changed_files,
            "ingested_updates": ingest_res,
        }

    def serialize(self) -> str:
        return json.dumps(
            {
                "chunks": [asdict(c) for c in self.chunks],
                "file_state": self.file_state,
            }
        )


INDEX = RepositoryIndex()
