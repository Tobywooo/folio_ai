"""Clients for the Llama chat and embedding APIs (OpenAI-compatible request/response format)."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .config import Settings
from .errors import ModelAPIError

log = logging.getLogger(__name__)

_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4


def _post(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    last_error = ""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if resp.status_code not in _RETRY_STATUSES:
                if not resp.ok:
                    raise ModelAPIError(f"{url} returned HTTP {resp.status_code}: {resp.text[:500]}")
                try:
                    return resp.json()
                except ValueError as exc:
                    raise ModelAPIError(f"{url} returned a non-JSON response.") from exc
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit() and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(min(int(retry_after), 30))
                continue
        if attempt < _MAX_ATTEMPTS - 1:
            delay = 2**attempt
            log.warning("Model API call to %s failed (%s); retrying in %ss", url, last_error, delay)
            time.sleep(delay)
    raise ModelAPIError(f"{url} failed after {_MAX_ATTEMPTS} attempts: {last_error}")


def chat(settings: Settings, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
    data = _post(
        f"{settings.llm_base_url}/chat/completions",
        settings.llm_api_key,
        {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        },
        timeout=180,
    )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelAPIError(f"Unexpected chat response shape: {str(data)[:300]}") from exc
    if isinstance(content, list):  # some servers return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return (content or "").strip()


def embed(settings: Settings, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start : start + settings.embedding_batch_size]
        data = _post(
            f"{settings.embedding_base_url}/embeddings",
            settings.embedding_api_key,
            {"model": settings.embedding_model, "input": batch},
            timeout=120,
        )
        try:
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            batch_vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise ModelAPIError(f"Unexpected embeddings response shape: {str(data)[:300]}") from exc
        if len(batch_vectors) != len(batch):
            raise ModelAPIError(f"Embeddings API returned {len(batch_vectors)} vectors for {len(batch)} inputs.")
        for vector in batch_vectors:
            if len(vector) != settings.embedding_dimensions:
                raise ModelAPIError(
                    f"Embedding model returned {len(vector)}-dimensional vectors but EMBEDDING_DIMENSIONS is "
                    f"{settings.embedding_dimensions}. Update EMBEDDING_DIMENSIONS and use a new AZURE_SEARCH_INDEX_NAME."
                )
        vectors.extend(batch_vectors)
    return vectors
