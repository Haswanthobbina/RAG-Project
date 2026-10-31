"""Configuration settings for GitContext / RAG-Project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _load_env_file() -> None:
    """Loads key-value pairs from .env into os.environ if not already set."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass


_load_env_file()


@dataclass
class Settings:
    APP_NAME: str = "GitContext: Codebase Intelligence Copilot"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Ingestion Settings
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".js",
        ".ts",
        ".go",
        ".rs",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".sql",
    )
    IGNORED_DIRS: Tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
    )

    # Retrieval Tuning
    HYBRID_ALPHA: float = 0.5  # 0.0 = Pure BM25, 1.0 = Pure Vector, 0.5 = Balanced
    RRF_K: int = 60            # Reciprocal Rank Fusion constant
    TOP_K_RETRIEVAL: int = 20  # Candidates fetched before reranking
    TOP_K_RERANK: int = 5      # Final returned candidates


settings = Settings()
