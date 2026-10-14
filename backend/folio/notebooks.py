"""Notebook lifecycle. Deleting a notebook removes its blobs, index chunks, and table rows."""

from __future__ import annotations

from typing import Any

from .context import Context
from .errors import ValidationError
from .storage import check_id

_MAX_NAME = 100


def clean_name(name: Any) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise ValidationError("Notebook name can't be empty.")
    if len(name) > _MAX_NAME:
        raise ValidationError(f"Notebook name must be {_MAX_NAME} characters or fewer.")
    return name


def create(ctx: Context, name: Any) -> dict[str, Any]:
    return ctx.store.create_notebook(clean_name(name))


def rename(ctx: Context, notebook_id: str, name: Any) -> dict[str, Any]:
    return ctx.store.rename_notebook(notebook_id, clean_name(name))


def delete(ctx: Context, notebook_id: str) -> None:
    check_id(notebook_id, "notebook id")
    ctx.store.get_notebook(notebook_id)
    ctx.index.delete_notebook(notebook_id)
    for container in ctx.store.all_containers():
        ctx.store.delete_prefix(container, f"{notebook_id}/")
    ctx.store.delete_notebook_record(notebook_id)
