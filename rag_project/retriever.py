"""Hybrid retriever combining Vector Search, BM25 Keyword Search, RRF, and Reranking."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from rag_project.bm25_index import BM25SearchEngine
from rag_project.config import settings
from rag_project.vector_store import VectorStore


class HybridRetriever:
    """Orchestrates Hybrid Retrieval using Reciprocal Rank Fusion (RRF) and Reranking."""

    def __init__(self, vector_store: VectorStore, bm25_engine: BM25SearchEngine) -> None:
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine

    @staticmethod
    def _chunk_unique_key(chunk: Dict[str, Any]) -> str:
        return f"{chunk.get('file_path', '')}::{chunk.get('start_line', 0)}::{chunk.get('name', '')}"

    @staticmethod
    def rerank_score(query: str, chunk: Dict[str, Any]) -> float:
        """Cross-Encoder style precision scoring based on exact match, symbol coverage, and title proximity."""
        q_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) > 1]
        if not q_tokens:
            return 0.0

        content = chunk.get("text", "").lower()
        name = chunk.get("name", "").lower()
        file_path = chunk.get("file_path", "").lower()

        overlap_content = sum(1 for t in q_tokens if t in content)
        overlap_name = sum(2 for t in q_tokens if t in name)
        overlap_file = sum(1.5 for t in q_tokens if t in file_path)

        # Exact phrase or function signature bonus
        phrase_bonus = 0.5 if query.lower().strip() in content else 0.0

        score = (overlap_content + overlap_name + overlap_file) / (len(q_tokens) * 3.0) + phrase_bonus
        return min(1.0, score)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = settings.HYBRID_ALPHA,
        rrf_k: int = settings.RRF_K,
    ) -> List[Dict[str, Any]]:
        """Executes hybrid retrieval:
        1. Fetch candidates from Vector Store & BM25
        2. Reciprocal Rank Fusion (RRF)
        3. Cross-Encoder / Precision Reranking
        """
        vector_hits = self.vector_store.search(query, top_k=settings.TOP_K_RETRIEVAL)
        bm25_hits = self.bm25_engine.search(query, top_k=settings.TOP_K_RETRIEVAL)

        if not vector_hits and not bm25_hits:
            return []

        # RRF dictionary
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        vector_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        # 1. RRF on vector results
        for rank, (chunk, score) in enumerate(vector_hits):
            cid = self._chunk_unique_key(chunk)
            chunk_map[cid] = chunk
            vector_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (alpha / (rrf_k + rank + 1))

        # 2. RRF on BM25 results
        for rank, (chunk, score) in enumerate(bm25_hits):
            cid = self._chunk_unique_key(chunk)
            chunk_map[cid] = chunk
            bm25_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + ((1.0 - alpha) / (rrf_k + rank + 1))

        # 3. Shortlist candidates by RRF score
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[
            : settings.TOP_K_RETRIEVAL
        ]

        # 4. Reranking step
        final_results = []
        for cid in sorted_candidates:
            chunk = chunk_map[cid]
            rrf_s = rrf_scores[cid]
            rerank_s = self.rerank_score(query, chunk)
            v_s = vector_scores.get(cid, 0.0)
            b_s = bm25_scores.get(cid, 0.0)

            # Combined score: RRF + Reranker boost
            composite_score = (rrf_s * 10.0) + (rerank_s * 0.5)

            final_results.append(
                {
                    "chunk_id": chunk.get("chunk_id", cid),
                    "score": round(composite_score, 4),
                    "rrf_score": round(rrf_s, 6),
                    "vector_score": round(v_s, 4),
                    "bm25_score": round(b_s, 4),
                    "rerank_score": round(rerank_s, 4),
                    "metadata": chunk.get("metadata", {}),
                    "text": chunk.get("text", ""),
                }
            )

        final_results.sort(key=lambda item: item["score"], reverse=True)
        return final_results[:top_k]
