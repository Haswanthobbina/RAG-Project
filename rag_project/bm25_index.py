"""BM25 keyword search engine for code and technical documentation."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore[assignment]


# Code-aware tokenization regex
CODE_SPLIT_REGEX = re.compile(r"([a-z0-9])([A-Z])|([A-Z]+)([A-Z][a-z])")
WORD_REGEX = re.compile(r"[A-Za-z0-9_#\-\.:]+")


class BM25SearchEngine:
    """Keyword search engine with code-aware tokenization."""

    def __init__(self) -> None:
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_model: Any = None
        self.doc_freq: Dict[str, int] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize code text, splitting camelCase and snake_case while preserving original identifiers."""
        if not text:
            return []

        # Convert CamelCase to separate words: MySuperClass -> My Super Class
        spaced = CODE_SPLIT_REGEX.sub(r"\1 \2\3 \4", text)
        raw_words = WORD_REGEX.findall(spaced)

        tokens: List[str] = []
        for word in raw_words:
            lowered = word.lower()
            tokens.append(lowered)

            # Sub-split on underscore or dot for symbols like `auth_service.py` -> `auth`, `service`, `py`
            parts = re.split(r"[_\-\.]", lowered)
            if len(parts) > 1:
                for p in parts:
                    if len(p) > 1 and p not in tokens:
                        tokens.append(p)

        return tokens

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes all chunk texts and associated file/function metadata."""
        self.corpus_chunks = chunks
        self.tokenized_corpus = []

        for c in chunks:
            text_to_index = f"{c.get('file_path', '')} {c.get('name', '')} {c.get('text', '')}"
            self.tokenized_corpus.append(self.tokenize(text_to_index))

        if BM25Okapi is not None and self.tokenized_corpus:
            self.bm25_model = BM25Okapi(self.tokenized_corpus)
        else:
            # Fallback pure-Python BM25 computation
            self.doc_freq = {}
            for doc in self.tokenized_corpus:
                for token in set(doc):
                    self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """Searches BM25 index and returns list of (chunk, score)."""
        if not self.corpus_chunks or not self.tokenized_corpus:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        if self.bm25_model is not None:
            scores = self.bm25_model.get_scores(query_tokens)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results: List[Tuple[Dict[str, Any], float]] = []
            for idx in sorted_indices:
                score = float(scores[idx])
                if score > 0.0:
                    results.append((self.corpus_chunks[idx], score))
            return results

        # Fallback pure-python BM25
        n_docs = len(self.corpus_chunks)
        results = []
        for i, doc in enumerate(self.tokenized_corpus):
            doc_len = len(doc)
            if doc_len == 0:
                continue
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1

            score = 0.0
            for t in query_tokens:
                if t in tf:
                    df = self.doc_freq.get(t, 1)
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    term_score = (tf[t] * 2.2) / (tf[t] + 1.2 * (0.25 + 0.75 * (doc_len / 50.0)))
                    score += idf * term_score

            if score > 0.0:
                results.append((self.corpus_chunks[i], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
