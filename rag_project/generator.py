"""LLM Copilot answer generator with grounded citations."""
from __future__ import annotations

import os
from typing import Any, Dict, List
from rag_project.config import settings


class CodeCopilotGenerator:
    """Synthesizes code explanations, root-cause analyses, and answers with file/line/author citations."""

    def __init__(self) -> None:
        self.client = None
        self.provider = "built-in"

        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai

                genai.configure(api_key=settings.GEMINI_API_KEY)
                
                # Try contemporary models in priority order
                candidate_models = [
                    "gemini-3.8-flash",
                    "gemini-flash-latest",
                    "gemini-2.5-flash",
                    "gemini-1.5-flash",
                ]
                for model_name in candidate_models:
                    try:
                        self.client = genai.GenerativeModel(model_name)
                        self.provider = f"gemini ({model_name})"
                        break
                    except Exception:
                        continue
            except Exception:
                pass

    def build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        snippets = []
        for i, c in enumerate(contexts):
            meta = c.get("metadata", {})
            file_p = meta.get("file_path", "unknown")
            start = meta.get("start_line", "1")
            end = meta.get("end_line", "1")
            author = meta.get("author", "N/A")
            commit = meta.get("last_commit_hash", "N/A")
            text = c.get("text", "")

            block = (
                f"### Context Block {i+1}:\n"
                f"- File: {file_p} (Lines {start}-{end})\n"
                f"- Author: {author} (Commit: {commit})\n"
                f"```\n{text}\n```\n"
            )
            snippets.append(block)

        joined_snippets = "\n".join(snippets)

        return f"""You are GitContext, a high-precision Codebase Intelligence Copilot.
Answer the developer's question using ONLY the provided code snippets and metadata.
Strictly cite your sources using the format `[filename:start_line-end_line]`.

Question:
{question}

Repository Context:
{joined_snippets}

Guidelines:
1. Explain the architecture, function, or bug directly and clearly.
2. Provide code snippets if offering a fix or example.
3. Attribute authors, commits, or PR references if relevant.
4. If the provided context is insufficient, state exactly what is missing.
"""

    def generate(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates answer and structured citations."""
        citations = []
        for c in contexts:
            meta = c.get("metadata", {})
            citations.append(
                {
                    "file_path": meta.get("file_path", "unknown"),
                    "lines": f"{meta.get('start_line', 1)}-{meta.get('end_line', 1)}",
                    "chunk_type": meta.get("chunk_type", "general"),
                    "name": meta.get("name", "block"),
                    "author": meta.get("author", "unknown"),
                    "commit": meta.get("last_commit_hash", "N/A"),
                    "score": c.get("score", 0.0),
                }
            )

        if not contexts:
            return {
                "question": question,
                "answer": "No relevant code or documentation was found in the indexed repository for this query.",
                "citations": [],
            }

        if self.client is not None and "gemini" in self.provider:
            try:
                prompt = self.build_prompt(question, contexts)
                resp = self.client.generate_content(prompt)
                return {
                    "question": question,
                    "answer": resp.text,
                    "citations": citations,
                    "provider": self.provider,
                }
            except Exception as e:
                # Fallback to high-fidelity built-in generator if network/quota error
                pass

        # High-Fidelity Built-in Synthesizer
        best_match = contexts[0]
        meta_best = best_match.get("metadata", {})
        top_name = meta_best.get("name", "target symbol")
        top_file = meta_best.get("file_path", "unknown")
        top_lines = f"{meta_best.get('start_line', 1)}-{meta_best.get('end_line', 1)}"
        top_author = meta_best.get("author", "local_developer")
        top_commit = meta_best.get("last_commit_hash", "untracked")

        summary_lines = [
            f"### ⚡ Analysis for: `{question}`",
            "",
            f"Found **{len(contexts)} relevant code segments** matching your query. The most relevant definition is **`{top_name}`** in `[{top_file}:{top_lines}]` (authored by **{top_author}**, commit `{top_commit}`).",
            "",
            "#### Primary Matched Code:",
            f"```python\n# File: {top_file}:{top_lines}\n{best_match.get('text', '').strip()}\n```",
            "",
            "#### Context Breakdown & Cross References:",
        ]

        for i, c in enumerate(contexts):
            m = c.get("metadata", {})
            summary_lines.append(
                f"- **[{m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')}]** "
                f"(`{m.get('chunk_type', 'code')}`: `{m.get('name')}`) — "
                f"Hybrid Score: `{c.get('score')}` (Vector: `{c.get('vector_score')}`, BM25: `{c.get('bm25_score')}`)"
            )

        return {
            "question": question,
            "answer": "\n".join(summary_lines),
            "citations": citations,
            "provider": "built-in-synthesizer",
        }
