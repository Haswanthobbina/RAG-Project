from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


class RepositoryIndex:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.chunk_vectors: List[Dict[str, float]] = []
        self.doc_freq: Dict[str, int] = {}
        self.file_state: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _tfidf(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        num_docs = max(1, len(self.chunks))
        vec: Dict[str, float] = {}
        for token, count in tf.items():
            df = self.doc_freq.get(token, 1)
            idf = math.log((num_docs + 1) / df)
            vec[token] = (count / len(tokens)) * idf
        return vec

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(val * b.get(key, 0.0) for key, val in a.items())
        a_mag = math.sqrt(sum(v * v for v in a.values()))
        b_mag = math.sqrt(sum(v * v for v in b.values()))
        if a_mag == 0 or b_mag == 0:
            return 0.0
        return dot / (a_mag * b_mag)

    @staticmethod
    def _bm25_like(query_tokens: List[str], text: str) -> float:
        text_tokens = RepositoryIndex._tokenize(text)
        if not text_tokens:
            return 0.0
        token_count: Dict[str, int] = {}
        for token in text_tokens:
            token_count[token] = token_count.get(token, 0) + 1
        return sum(math.log(1 + token_count.get(token, 0)) for token in query_tokens)

    @staticmethod
    def _rerank(query_tokens: List[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = set(RepositoryIndex._tokenize(text))
        overlap = sum(1 for token in query_tokens if token in text_tokens)
        return overlap / len(query_tokens)

    @staticmethod
    def _chunk_python(path: Path, content: str) -> Iterable[Tuple[str, str, int, int]]:
        parsed = ast.parse(content)
        lines = content.splitlines()

        for node in parsed.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", node.lineno)
                chunk_text = "\n".join(lines[start - 1 : end])
                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
                chunk_name = node.name
                yield chunk_type, chunk_name, start, end, chunk_text

        if parsed.body:
            first = parsed.body[0].lineno
            if first > 1:
                header = "\n".join(lines[: first - 1]).strip()
                if header:
                    yield "module", path.stem, 1, first - 1, header

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_chunk(self, path: Path, chunk_type: str, name: str, start: int, end: int, text: str) -> Chunk:
        chunk_id = f"{path.as_posix()}::{chunk_type}::{name}::{start}-{end}"
        metadata = {
            "file_path": path.as_posix(),
            "chunk_type": chunk_type,
            "name": name,
            "start_line": str(start),
            "end_line": str(end),
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        return Chunk(chunk_id=chunk_id, text=text, metadata=metadata)

    def ingest_repository(self, repo_path: str, include_extensions: Tuple[str, ...] = (".py", ".md", ".txt")) -> Dict[str, List[str]]:
        allowed_root = Path.cwd().resolve()
        if repo_path not in {allowed_root.as_posix(), str(allowed_root)}:
            raise ValueError(f"repo_path must match the current workspace root: {allowed_root}")
        root = allowed_root
        added: List[str] = []
        updated: List[str] = []
        removed: List[str] = []

        current_files = {
            p.as_posix(): p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in include_extensions and ".git/" not in p.as_posix()
        }

        for tracked_file in list(self.file_state.keys()):
            if tracked_file not in current_files:
                removed.append(tracked_file)
                self.file_state.pop(tracked_file, None)
                self.chunks = [chunk for chunk in self.chunks if chunk.metadata["file_path"] != tracked_file]

        for rel_path, absolute_path in current_files.items():
            text = absolute_path.read_text(encoding="utf-8", errors="ignore")
            digest = self._hash_text(text)
            mtime = str(absolute_path.stat().st_mtime_ns)
            state = self.file_state.get(rel_path)
            if state and state["digest"] == digest and state["mtime"] == mtime:
                continue

            if state:
                updated.append(rel_path)
            else:
                added.append(rel_path)

            self.chunks = [chunk for chunk in self.chunks if chunk.metadata["file_path"] != rel_path]

            if absolute_path.suffix.lower() == ".py":
                try:
                    produced_any = False
                    for chunk_type, name, start, end, chunk_text in self._chunk_python(absolute_path, text):
                        self.chunks.append(self._build_chunk(absolute_path, chunk_type, name, start, end, chunk_text))
                        produced_any = True
                    if not produced_any:
                        self.chunks.append(self._build_chunk(absolute_path, "module", absolute_path.stem, 1, max(1, len(text.splitlines())), text))
                except SyntaxError:
                    self.chunks.append(self._build_chunk(absolute_path, "module", absolute_path.stem, 1, max(1, len(text.splitlines())), text))
            else:
                self.chunks.append(self._build_chunk(absolute_path, "document", absolute_path.stem, 1, max(1, len(text.splitlines())), text))

            self.file_state[rel_path] = {"digest": digest, "mtime": mtime}

        self._recompute_statistics()

        return {"added": added, "updated": updated, "removed": removed}

    def _recompute_statistics(self) -> None:
        self.doc_freq = {}
        for chunk in self.chunks:
            unique_tokens = set(self._tokenize(chunk.text))
            for token in unique_tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.chunk_vectors = [self._tfidf(chunk.text) for chunk in self.chunks]

    def query(self, prompt: str, top_k: int = 5) -> List[Dict[str, object]]:
        query_tokens = self._tokenize(prompt)
        query_vector = self._tfidf(prompt)

        scored: List[Tuple[float, Chunk]] = []
        for chunk, vec in zip(self.chunks, self.chunk_vectors):
            vector_score = self._cosine_similarity(query_vector, vec)
            keyword_score = self._bm25_like(query_tokens, chunk.text)
            combined = 0.6 * vector_score + 0.4 * keyword_score
            scored.append((combined, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        shortlisted = scored[: max(top_k * 2, top_k)]

        reranked = []
        for score, chunk in shortlisted:
            rerank_score = self._rerank(query_tokens, chunk.text)
            reranked.append((score + 0.2 * rerank_score, score, rerank_score, chunk))

        reranked.sort(key=lambda item: item[0], reverse=True)
        results = []
        for final_score, hybrid_score, rerank_score, chunk in reranked[:top_k]:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": round(final_score, 6),
                    "hybrid_score": round(hybrid_score, 6),
                    "rerank_score": round(rerank_score, 6),
                    "metadata": chunk.metadata,
                    "text": chunk.text,
                }
            )
        return results

    def serialize(self) -> str:
        return json.dumps(
            {
                "chunks": [asdict(chunk) for chunk in self.chunks],
                "file_state": self.file_state,
            }
        )


INDEX = RepositoryIndex()
