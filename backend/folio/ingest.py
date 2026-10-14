"""Source ingestion: store raw file -> extract text -> chunk -> embed -> index -> store enriched JSON."""

from __future__ import annotations

import json
import logging
import re
from pathlib import PurePath
from typing import Any

from . import model_api
from .chunking import chunk_sections
from .context import Context
from .errors import FolioError, IngestionError, ValidationError
from .extract import Section, extract_pdf, extract_pptx, extract_txt, fetch_url
from .storage import check_id, new_id, now_iso

log = logging.getLogger(__name__)

_EXTENSIONS = {".pdf": "pdf", ".pptx": "pptx", ".txt": "txt"}
_CONTENT_TYPES = {
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt": "text/plain; charset=utf-8",
    "url": "text/html",
}
_EXTRACTORS = {"pdf": extract_pdf, "pptx": extract_pptx, "txt": extract_txt}


def safe_filename(name: str) -> str:
    base = PurePath(name.replace("\\", "/")).name
    base = re.sub(r"[^\w.\- ]+", "_", base).strip(" .")
    return base[:150] or "upload"


def source_type_for(filename: str) -> str:
    source_type = _EXTENSIONS.get(PurePath(filename).suffix.lower())
    if not source_type:
        raise ValidationError("Unsupported file type. Upload a .pdf, .pptx, or .txt file.")
    return source_type


def ingest_file(ctx: Context, notebook_id: str, filename: str, data: bytes) -> dict[str, Any]:
    ctx.store.get_notebook(notebook_id)
    filename = safe_filename(filename)
    source_type = source_type_for(filename)
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > ctx.settings.max_upload_bytes:
        raise ValidationError(f"File is larger than {ctx.settings.max_upload_bytes // (1024 * 1024)} MB.")

    source_id = new_id()
    raw_container, _ = ctx.settings.containers[source_type]
    raw_blob = f"{notebook_id}/{source_id}/{filename}"
    _record(ctx, notebook_id, source_id, name=filename, type=source_type, status="processing",
            raw_container=raw_container, raw_blob=raw_blob, url="", created_at=now_iso())
    ctx.store.put_blob(raw_container, raw_blob, data, _CONTENT_TYPES[source_type])

    return _process(ctx, notebook_id, source_id, filename, source_type, "", lambda: _EXTRACTORS[source_type](data))


def ingest_url(ctx: Context, notebook_id: str, url: str) -> dict[str, Any]:
    ctx.store.get_notebook(notebook_id)
    url = (url or "").strip()
    if not url:
        raise ValidationError("Enter a URL.")

    source_id = new_id()
    raw_container, _ = ctx.settings.containers["url"]
    _record(ctx, notebook_id, source_id, name=url, type="url", status="processing",
            raw_container=raw_container, raw_blob="", url=url, created_at=now_iso())

    def extract() -> list[Section]:
        page = fetch_url(url)
        raw_blob = f"{notebook_id}/{source_id}/page.html"
        ctx.store.put_blob(raw_container, raw_blob, page.html, _CONTENT_TYPES["url"])
        _record(ctx, notebook_id, source_id, name=page.title[:200], url=page.url, raw_blob=raw_blob)
        return page.sections

    return _process(ctx, notebook_id, source_id, url, "url", url, extract)


def delete_source(ctx: Context, notebook_id: str, source_id: str) -> None:
    check_id(notebook_id, "notebook id")
    source = ctx.store.get_source(notebook_id, source_id)
    ctx.index.delete_source(source_id)
    raw_container, enriched_container = ctx.settings.containers[source["type"]]
    ctx.store.delete_prefix(raw_container, f"{notebook_id}/{source_id}/")
    ctx.store.delete_prefix(enriched_container, f"{notebook_id}/{source_id}.json")
    ctx.store.delete_source_record(notebook_id, source_id)


# ---------------------------------------------------------------------------


def _record(ctx: Context, notebook_id: str, source_id: str, **fields: Any) -> None:
    ctx.store.upsert_source(notebook_id, source_id, fields)


def _process(ctx, notebook_id, source_id, display_name, source_type, url, extract) -> dict[str, Any]:
    settings = ctx.settings
    try:
        sections = extract()
        chunks = chunk_sections(sections, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            hint = " Scanned/image-only PDFs need OCR, which isn't supported." if source_type == "pdf" else ""
            raise IngestionError(f"No text could be extracted from this source.{hint}")

        source = ctx.store.get_source(notebook_id, source_id)
        name = source.get("name") or display_name
        vectors = model_api.embed(settings, [c.text for c in chunks])
        documents = [
            {
                "id": f"{source_id}-{chunk.index:05d}",
                "notebook_id": notebook_id,
                "source_id": source_id,
                "source_type": source_type,
                "source_name": name,
                "source_url": source.get("url") or url,
                "page": chunk.page,
                "chunk_index": chunk.index,
                "content": chunk.text,
                "content_vector": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        ctx.index.upload(documents)

        _, enriched_container = settings.containers[source_type]
        enriched_blob = f"{notebook_id}/{source_id}.json"
        enriched = {
            "notebook_id": notebook_id,
            "source_id": source_id,
            "source_type": source_type,
            "source_name": name,
            "source_url": source.get("url") or url,
            "raw_blob": f"{source.get('raw_container')}/{source.get('raw_blob')}",
            "created_at": now_iso(),
            "chunking": {"strategy": "recursive-character", "size": settings.chunk_size, "overlap": settings.chunk_overlap},
            "embedding_model": settings.embedding_model,
            "sections": [{"page": s.page, "text": s.text} for s in sections],
            "chunks": [{"id": d["id"], "page": d["page"], "chunk_index": d["chunk_index"], "text": d["content"]} for d in documents],
        }
        ctx.store.put_blob(enriched_container, enriched_blob, json.dumps(enriched, ensure_ascii=False).encode("utf-8"),
                           "application/json")

        _record(ctx, notebook_id, source_id, status="ready", chunk_count=len(chunks), error="",
                enriched_container=enriched_container, enriched_blob=enriched_blob)
        return ctx.store.get_source(notebook_id, source_id)
    except FolioError as exc:
        _fail(ctx, notebook_id, source_id, str(exc))
        raise
    except Exception as exc:
        log.exception("Ingestion failed for source %s", source_id)
        _fail(ctx, notebook_id, source_id, f"Unexpected error: {exc}")
        raise IngestionError(f"Ingestion failed: {exc}") from exc


def _fail(ctx: Context, notebook_id: str, source_id: str, message: str) -> None:
    try:
        ctx.index.delete_source(source_id)  # drop any partially uploaded chunks
        _record(ctx, notebook_id, source_id, status="failed", error=message[:1000])
    except Exception:
        log.exception("Could not mark source %s as failed", source_id)
