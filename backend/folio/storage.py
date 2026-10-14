"""Azure Blob + Table storage access.

Layout
------
Blobs:   <raw container>/<notebook_id>/<source_id>/<filename>
         <enriched container>/<notebook_id>/<source_id>.json
         <artifacts container>/<notebook_id>/<artifact_id>.md
Tables:  {prefix}notebooks  PK="notebooks"     RK=notebook_id
         {prefix}sources    PK=notebook_id     RK=source_id
         {prefix}messages   PK=notebook_id     RK=<ns timestamp>-<rand>   (sorts chronologically)
         {prefix}artifacts  PK=notebook_id     RK=artifact_id
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from azure.storage.blob import BlobServiceClient, ContentSettings

from .config import Settings
from .errors import NotFoundError, ValidationError

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_NOTEBOOKS_PK = "notebooks"
# Table Storage caps string properties at 64 KiB (32K UTF-16 chars).
_MAX_TABLE_STRING = 30_000


def new_id() -> str:
    return uuid.uuid4().hex


def check_id(value: str, what: str = "id") -> str:
    """IDs end up in blob paths, table keys, and OData filters, so only accept our own format."""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValidationError(f"Invalid {what}.")
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(entity: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in entity.items() if k not in ("PartitionKey", "RowKey")}
    out["id"] = entity["RowKey"]
    return out


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._blobs = BlobServiceClient.from_connection_string(settings.storage_connection_string)
        self._tables = TableServiceClient.from_connection_string(settings.storage_connection_string)
        self.notebooks = self._tables.get_table_client(settings.table_name("notebooks"))
        self.sources = self._tables.get_table_client(settings.table_name("sources"))
        self.messages = self._tables.get_table_client(settings.table_name("messages"))
        self.artifacts = self._tables.get_table_client(settings.table_name("artifacts"))

    # ------------------------------------------------------------------ setup

    def all_containers(self) -> list[str]:
        names = [name for pair in self.settings.containers.values() for name in pair]
        return names + [self.settings.artifacts_container]

    def ensure_resources(self) -> None:
        for name in self.all_containers():
            try:
                self._blobs.create_container(name)
            except ResourceExistsError:
                pass
        for kind in ("notebooks", "sources", "messages", "artifacts"):
            self._tables.create_table_if_not_exists(self.settings.table_name(kind))

    # ------------------------------------------------------------------ blobs

    def put_blob(self, container: str, path: str, data: bytes, content_type: str) -> None:
        self._blobs.get_container_client(container).upload_blob(
            path, data, overwrite=True, content_settings=ContentSettings(content_type=content_type)
        )

    def get_blob(self, container: str, path: str) -> bytes:
        try:
            return self._blobs.get_container_client(container).download_blob(path).readall()
        except ResourceNotFoundError as exc:
            raise NotFoundError("File not found in storage.") from exc

    def delete_prefix(self, container: str, prefix: str) -> int:
        client = self._blobs.get_container_client(container)
        count = 0
        for blob in client.list_blobs(name_starts_with=prefix):
            try:
                client.delete_blob(blob.name, delete_snapshots="include")
                count += 1
            except ResourceNotFoundError:
                pass
        return count

    # -------------------------------------------------------------- notebooks

    def list_notebooks(self) -> list[dict[str, Any]]:
        rows = self.notebooks.query_entities("PartitionKey eq @pk", parameters={"pk": _NOTEBOOKS_PK})
        return sorted((_clean(r) for r in rows), key=lambda n: n.get("created_at", ""), reverse=True)

    def get_notebook(self, notebook_id: str) -> dict[str, Any]:
        check_id(notebook_id, "notebook id")
        try:
            return _clean(self.notebooks.get_entity(_NOTEBOOKS_PK, notebook_id))
        except ResourceNotFoundError as exc:
            raise NotFoundError("Notebook not found.") from exc

    def create_notebook(self, name: str) -> dict[str, Any]:
        ts = now_iso()
        entity = {"PartitionKey": _NOTEBOOKS_PK, "RowKey": new_id(), "name": name, "created_at": ts, "updated_at": ts}
        self.notebooks.create_entity(entity)
        return _clean(entity)

    def rename_notebook(self, notebook_id: str, name: str) -> dict[str, Any]:
        self.get_notebook(notebook_id)
        self.notebooks.update_entity(
            {"PartitionKey": _NOTEBOOKS_PK, "RowKey": notebook_id, "name": name, "updated_at": now_iso()},
            mode=UpdateMode.MERGE,
        )
        return self.get_notebook(notebook_id)

    def delete_notebook_record(self, notebook_id: str) -> None:
        for table in (self.sources, self.messages, self.artifacts):
            self._delete_partition(table, notebook_id)
        try:
            self.notebooks.delete_entity(_NOTEBOOKS_PK, notebook_id)
        except ResourceNotFoundError:
            pass

    def _delete_partition(self, table, partition_key: str) -> None:
        rows = table.query_entities("PartitionKey eq @pk", parameters={"pk": partition_key}, select=["RowKey"])
        for row in rows:
            try:
                table.delete_entity(partition_key, row["RowKey"])
            except ResourceNotFoundError:
                pass

    # ---------------------------------------------------------------- sources

    def upsert_source(self, notebook_id: str, source_id: str, fields: dict[str, Any]) -> None:
        self.sources.upsert_entity({"PartitionKey": notebook_id, "RowKey": source_id, **fields}, mode=UpdateMode.MERGE)

    def get_source(self, notebook_id: str, source_id: str) -> dict[str, Any]:
        check_id(source_id, "source id")
        try:
            return _clean(self.sources.get_entity(notebook_id, source_id))
        except ResourceNotFoundError as exc:
            raise NotFoundError("Source not found.") from exc

    def list_sources(self, notebook_id: str) -> list[dict[str, Any]]:
        rows = self.sources.query_entities("PartitionKey eq @pk", parameters={"pk": notebook_id})
        return sorted((_clean(r) for r in rows), key=lambda s: s.get("created_at", ""))

    def delete_source_record(self, notebook_id: str, source_id: str) -> None:
        try:
            self.sources.delete_entity(notebook_id, source_id)
        except ResourceNotFoundError:
            pass

    # --------------------------------------------------------------- messages

    def add_message(
        self, notebook_id: str, role: str, content: str, citations: list[dict[str, Any]] | None = None, mode: str = ""
    ) -> dict[str, Any]:
        citations_json = json.dumps(citations or [])
        if len(citations_json) > _MAX_TABLE_STRING:
            citations_json = json.dumps([{k: v for k, v in c.items() if k != "snippet"} for c in citations or []])
        entity = {
            "PartitionKey": notebook_id,
            "RowKey": f"{time.time_ns():020d}-{uuid.uuid4().hex[:6]}",
            "role": role,
            "content": content[:_MAX_TABLE_STRING],
            "citations": citations_json,
            "mode": mode,
            "created_at": now_iso(),
        }
        self.messages.create_entity(entity)
        return _message_out(entity)

    def list_messages(self, notebook_id: str) -> list[dict[str, Any]]:
        rows = self.messages.query_entities("PartitionKey eq @pk", parameters={"pk": notebook_id})
        return [_message_out(r) for r in sorted(rows, key=lambda r: r["RowKey"])]

    # -------------------------------------------------------------- artifacts

    def add_artifact(self, notebook_id: str, artifact_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        entity = {"PartitionKey": notebook_id, "RowKey": artifact_id, **fields}
        self.artifacts.create_entity(entity)
        return _clean(entity)

    def get_artifact(self, notebook_id: str, artifact_id: str) -> dict[str, Any]:
        check_id(artifact_id, "artifact id")
        try:
            return _clean(self.artifacts.get_entity(notebook_id, artifact_id))
        except ResourceNotFoundError as exc:
            raise NotFoundError("Artifact not found.") from exc

    def list_artifacts(self, notebook_id: str) -> list[dict[str, Any]]:
        rows = self.artifacts.query_entities("PartitionKey eq @pk", parameters={"pk": notebook_id})
        return sorted((_clean(r) for r in rows), key=lambda a: a.get("created_at", ""), reverse=True)


def _message_out(entity: dict[str, Any]) -> dict[str, Any]:
    out = _clean(entity)
    try:
        out["citations"] = json.loads(out.get("citations") or "[]")
    except json.JSONDecodeError:
        out["citations"] = []
    return out


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
