"""Centralized path and runtime configuration for Lever.

All paths can be overridden with environment variables so the repository
can be reproduced without hardcoding local experiment directories.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: str | Path) -> Path:
    """Return a filesystem path from an environment variable."""
    return Path(os.getenv(name, str(default))).expanduser()


DATA_ROOT = _path_from_env("RAG_DATA_ROOT", PROJECT_ROOT / "data")
MODEL_ROOT = _path_from_env("RAG_MODEL_ROOT", PROJECT_ROOT / "models")
ARTIFACT_ROOT = _path_from_env("RAG_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")

TAG_MODEL_PATH = _path_from_env("RAG_TAG_MODEL_PATH", MODEL_ROOT / "qwen-8b")
TAG_CACHE_PATH = _path_from_env("RAG_TAG_CACHE_PATH", ARTIFACT_ROOT / "tag.pkl")
DOMAIN_KNN_DIR = _path_from_env("RAG_DOMAIN_KNN_DIR", ARTIFACT_ROOT / "domain_knn")
OUTPUT_ROOT = _path_from_env("RAG_OUTPUT_ROOT", ARTIFACT_ROOT)
