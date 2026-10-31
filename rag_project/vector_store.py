"""Dense vector store and similarity search engine."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class VectorStore:
    """Vector index supporting dense cosine similarity with subword and semantic representations."""

    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: List[Dict[str, float]] = []
        self.doc_freq: Dict[str, int] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = [t.lower() for t in TOKEN_RE.findall(text)]
        # Add character 3-grams for subword matching (e.g., typos or prefix/suffix matches)
        subwords = []
        for t in tokens:
            if len(t) >= 4:
                for i in range(len(t) - 2):
                    subwords.append(t[i : i + 3])
        return tokens + subwords

    def _compute_vector(self, text: str) -> Dict[str, float]:
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
            # Smooth IDF
            idf = math.log((num_docs + 1) / (df + 1)) + 1.0
            vec[token] = (count / len(tokens)) * idf

        # Normalize L2
        mag = math.sqrt(sum(v * v for v in vec.values()))
        if mag > 0:
            for k in vec:
                vec[k] /= mag
        return vec

    @staticmethod
    def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        # Dot product of L2 normalized vectors
        if len(a) > len(b):
            a, b = b, a
        return sum(val * b.get(key, 0.0) for key, val in a.items())

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes chunks and builds normalized vector representations."""
        self.chunks = chunks
        self.doc_freq = {}

        # 1. Calculate document frequencies
        for chunk in self.chunks:
            text = f"{chunk.get('name', '')} {chunk.get('file_path', '')} {chunk.get('text', '')}"
            unique_tokens = set(self._tokenize(text))
            for t in unique_tokens:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        # 2. Build normalized vectors
        self.vectors = []
        for chunk in self.chunks:
            text = f"{chunk.get('name', '')} {chunk.get('file_path', '')} {chunk.get('text', '')}"
            self.vectors.append(self._compute_vector(text))

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """Vector similarity search returning top-k matching (chunk, score)."""
        if not self.chunks or not self.vectors:
            return []

        query_vec = self._compute_vector(query)
        if not query_vec:
            return []

        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = self.cosine_similarity(query_vec, vec)
            if score > 0.0:
                scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
