"""Settings loaded from environment variables (plus the repo-root .env during local development)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

# Locally this picks up folio_ai/.env. In Azure the file doesn't exist and the values come
# from the Function App's environment variables. Existing env vars are never overridden.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SOURCE_TYPES = ("pdf", "pptx", "txt", "url")
RETRIEVAL_MODES = ("keyword", "vector", "hybrid", "hybrid_semantic")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable {name} (see .env.example).")
    return value


def _optional(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


@dataclass(frozen=True)
class Settings:
    storage_connection_string: str
    # source type -> (raw container, enriched container)
    containers: dict[str, tuple[str, str]]
    artifacts_container: str
    table_prefix: str

    search_endpoint: str
    search_api_key: str
    search_index_name: str

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimensions: int
    embedding_batch_size: int

    chunk_size: int
    chunk_overlap: int
    max_upload_bytes: int
    retrieval_mode: str
    retrieval_top_k: int
    chat_history_turns: int
    artifact_context_chars: int

    def table_name(self, kind: str) -> str:
        return f"{self.table_prefix}{kind}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    retrieval_mode = _optional("RETRIEVAL_MODE", "hybrid")
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ConfigError(f"RETRIEVAL_MODE must be one of {', '.join(RETRIEVAL_MODES)}.")

    chunk_size = _int("CHUNK_SIZE", 1200)
    chunk_overlap = _int("CHUNK_OVERLAP", 200)
    if chunk_size < 200 or not 0 <= chunk_overlap < chunk_size:
        raise ConfigError("CHUNK_SIZE must be >= 200 and CHUNK_OVERLAP must be between 0 and CHUNK_SIZE.")

    table_prefix = _optional("TABLE_PREFIX", "folio")
    if not table_prefix.isalnum() or not table_prefix[0].isalpha():
        raise ConfigError("TABLE_PREFIX must be letters/digits and start with a letter.")

    return Settings(
        storage_connection_string=_required("AZURE_STORAGE_CONNECTION_STRING"),
        containers={
            "pdf": (_optional("CONTAINER_PDF", "pdf-documents"), _optional("CONTAINER_PDF_ENRICHED", "pdf-documents-enriched")),
            "pptx": (_optional("CONTAINER_PPTX", "pptx-files"), _optional("CONTAINER_PPTX_ENRICHED", "pptx-files-enriched")),
            "txt": (_optional("CONTAINER_TXT", "txt-files"), _optional("CONTAINER_TXT_ENRICHED", "txt-files-enriched")),
            "url": (_optional("CONTAINER_WEB", "web-scraped"), _optional("CONTAINER_WEB_ENRICHED", "web-scraped-enriched")),
        },
        artifacts_container=_optional("CONTAINER_ARTIFACTS", "artifacts"),
        table_prefix=table_prefix,
        search_endpoint=_required("AZURE_SEARCH_ENDPOINT"),
        search_api_key=_required("AZURE_SEARCH_API_KEY"),
        search_index_name=_optional("AZURE_SEARCH_INDEX_NAME", "folio-chunks"),
        llm_base_url=_required("LLM_API_BASE_URL").rstrip("/"),
        llm_api_key=_required("LLM_API_KEY"),
        llm_model=_required("LLM_MODEL"),
        llm_temperature=_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_int("LLM_MAX_TOKENS", 2048),
        embedding_base_url=_required("EMBEDDING_API_BASE_URL").rstrip("/"),
        embedding_api_key=_required("EMBEDDING_API_KEY"),
        embedding_model=_required("EMBEDDING_MODEL"),
        embedding_dimensions=_int("EMBEDDING_DIMENSIONS", 0) or int(_required("EMBEDDING_DIMENSIONS")),
        embedding_batch_size=max(1, _int("EMBEDDING_BATCH_SIZE", 16)),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_upload_bytes=_int("MAX_UPLOAD_MB", 25) * 1024 * 1024,
        retrieval_mode=retrieval_mode,
        retrieval_top_k=max(1, _int("RETRIEVAL_TOP_K", 5)),
        chat_history_turns=max(0, _int("CHAT_HISTORY_TURNS", 6)),
        artifact_context_chars=max(2000, _int("ARTIFACT_CONTEXT_CHARS", 24000)),
    )
