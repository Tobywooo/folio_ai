"""Thin HTTP client for the Folio AI Function App API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class BackendError(Exception):
    """Error message suitable for showing in the UI."""


class Backend:
    def __init__(self, base_url: str, function_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if function_key:
            self.session.headers["x-functions-key"] = function_key

    def _request(self, method: str, path: str, timeout: float = 30, **kwargs: Any) -> Any:
        try:
            resp = self.session.request(method, f"{self.base_url}{path}", timeout=timeout, **kwargs)
        except requests.ConnectionError as exc:
            raise BackendError(f"Can't reach the backend at {self.base_url}. Is the Function App running?") from exc
        except requests.Timeout as exc:
            raise BackendError("The backend took too long to respond. Try again in a moment.") from exc
        if resp.status_code == 401:
            raise BackendError("The backend rejected the request (401). Check BACKEND_FUNCTION_KEY.")
        try:
            data = resp.json()
        except ValueError:
            data = None
        if not resp.ok:
            message = data.get("error") if isinstance(data, dict) else None
            raise BackendError(message or f"Backend error (HTTP {resp.status_code}).")
        return data

    # notebooks
    def list_notebooks(self) -> list[dict[str, Any]]:
        return self._request("GET", "/notebooks")

    def create_notebook(self, name: str) -> dict[str, Any]:
        return self._request("POST", "/notebooks", json={"name": name})

    def rename_notebook(self, notebook_id: str, name: str) -> dict[str, Any]:
        return self._request("PATCH", f"/notebooks/{notebook_id}", json={"name": name})

    def delete_notebook(self, notebook_id: str) -> None:
        self._request("DELETE", f"/notebooks/{notebook_id}", timeout=120)

    # sources
    def list_sources(self, notebook_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/notebooks/{notebook_id}/sources")

    def upload_file(self, notebook_id: str, path: str) -> dict[str, Any]:
        file = Path(path)
        return self._request(
            "POST",
            f"/notebooks/{notebook_id}/sources",
            params={"filename": file.name},
            data=file.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=300,
        )

    def add_url(self, notebook_id: str, url: str) -> dict[str, Any]:
        return self._request("POST", f"/notebooks/{notebook_id}/urls", json={"url": url}, timeout=300)

    def delete_source(self, notebook_id: str, source_id: str) -> None:
        self._request("DELETE", f"/notebooks/{notebook_id}/sources/{source_id}", timeout=60)

    # chat
    def list_messages(self, notebook_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/notebooks/{notebook_id}/messages")

    def chat(self, notebook_id: str, question: str, mode: str) -> dict[str, Any]:
        return self._request("POST", f"/notebooks/{notebook_id}/chat", json={"question": question, "mode": mode}, timeout=240)

    # artifacts
    def list_artifacts(self, notebook_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/notebooks/{notebook_id}/artifacts")

    def create_artifact(self, notebook_id: str, kind: str, instructions: str, question_count: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/notebooks/{notebook_id}/artifacts",
            json={"kind": kind, "instructions": instructions, "question_count": question_count},
            timeout=300,
        )

    def get_artifact(self, notebook_id: str, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/notebooks/{notebook_id}/artifacts/{artifact_id}")
