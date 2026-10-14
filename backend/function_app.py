"""Folio AI HTTP API (Azure Functions, Python v2 programming model).

All routes are under /api and require a function key (header `x-functions-key`) when deployed.

GET    /notebooks                                   list notebooks
POST   /notebooks                     {name}        create
PATCH  /notebooks/{nb}                {name}        rename
DELETE /notebooks/{nb}                              delete (blobs, index chunks, history, artifacts)
GET    /notebooks/{nb}/sources                      list sources
POST   /notebooks/{nb}/sources?filename=x.pdf       upload file (raw bytes body) and index it
POST   /notebooks/{nb}/urls           {url}         scrape a web page and index it
DELETE /notebooks/{nb}/sources/{src}                remove a source
GET    /notebooks/{nb}/messages                     chat history
POST   /notebooks/{nb}/chat           {question, mode?}
GET    /notebooks/{nb}/artifacts                    list artifacts
POST   /notebooks/{nb}/artifacts      {kind, instructions?, question_count?}
GET    /notebooks/{nb}/artifacts/{id}               artifact metadata + Markdown content
GET    /health                                      config/connectivity check
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

import azure.functions as func

from folio import artifacts, ingest, notebooks, rag
from folio.context import get_context
from folio.errors import FolioError, ValidationError

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
log = logging.getLogger("folio")
# The Azure SDKs log every HTTP request/response at INFO, which buries our own logs.
logging.getLogger("azure").setLevel(logging.WARNING)


def _json(body: Any, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(body, default=str), status_code=status, mimetype="application/json")


def _body(req: func.HttpRequest) -> dict[str, Any]:
    try:
        data = req.get_json()
    except ValueError as exc:
        raise ValidationError("Request body must be JSON.") from exc
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object.")
    return data


def api(handler: Callable[[func.HttpRequest], Any]) -> Callable[[func.HttpRequest], func.HttpResponse]:
    """Turn return values into JSON responses and exceptions into JSON errors with the right status."""

    @functools.wraps(handler)
    def wrapper(req: func.HttpRequest) -> func.HttpResponse:
        try:
            result = handler(req)
            return result if isinstance(result, func.HttpResponse) else _json(result)
        except FolioError as exc:
            if exc.status_code >= 500:
                log.error("%s: %s", type(exc).__name__, exc)
            return _json({"error": str(exc)}, exc.status_code)
        except Exception:
            log.exception("Unhandled error in %s", handler.__name__)
            return _json({"error": "Internal server error. Check the Function App logs."}, 500)

    return wrapper


# ------------------------------------------------------------------ notebooks


@app.route(route="notebooks", methods=["GET"])
@api
def list_notebooks(req: func.HttpRequest):
    return get_context().store.list_notebooks()


@app.route(route="notebooks", methods=["POST"])
@api
def create_notebook(req: func.HttpRequest):
    return _json(notebooks.create(get_context(), _body(req).get("name")), 201)


@app.route(route="notebooks/{notebook_id}", methods=["PATCH"])
@api
def rename_notebook(req: func.HttpRequest):
    return notebooks.rename(get_context(), req.route_params["notebook_id"], _body(req).get("name"))


@app.route(route="notebooks/{notebook_id}", methods=["DELETE"])
@api
def delete_notebook(req: func.HttpRequest):
    notebooks.delete(get_context(), req.route_params["notebook_id"])
    return {"deleted": True}


# -------------------------------------------------------------------- sources


@app.route(route="notebooks/{notebook_id}/sources", methods=["GET"])
@api
def list_sources(req: func.HttpRequest):
    ctx = get_context()
    notebook_id = req.route_params["notebook_id"]
    ctx.store.get_notebook(notebook_id)
    return ctx.store.list_sources(notebook_id)


@app.route(route="notebooks/{notebook_id}/sources", methods=["POST"])
@api
def upload_source(req: func.HttpRequest):
    filename = req.params.get("filename", "")
    if not filename:
        raise ValidationError("Missing ?filename= query parameter.")
    source = ingest.ingest_file(get_context(), req.route_params["notebook_id"], filename, req.get_body())
    return _json(source, 201)


@app.route(route="notebooks/{notebook_id}/urls", methods=["POST"])
@api
def add_url(req: func.HttpRequest):
    source = ingest.ingest_url(get_context(), req.route_params["notebook_id"], str(_body(req).get("url", "")))
    return _json(source, 201)


@app.route(route="notebooks/{notebook_id}/sources/{source_id}", methods=["DELETE"])
@api
def delete_source(req: func.HttpRequest):
    ingest.delete_source(get_context(), req.route_params["notebook_id"], req.route_params["source_id"])
    return {"deleted": True}


# ----------------------------------------------------------------------- chat


@app.route(route="notebooks/{notebook_id}/messages", methods=["GET"])
@api
def list_messages(req: func.HttpRequest):
    ctx = get_context()
    notebook_id = req.route_params["notebook_id"]
    ctx.store.get_notebook(notebook_id)
    return ctx.store.list_messages(notebook_id)


@app.route(route="notebooks/{notebook_id}/chat", methods=["POST"])
@api
def chat(req: func.HttpRequest):
    body = _body(req)
    return rag.chat(get_context(), req.route_params["notebook_id"], str(body.get("question", "")), body.get("mode") or None)


# ------------------------------------------------------------------ artifacts


@app.route(route="notebooks/{notebook_id}/artifacts", methods=["GET"])
@api
def list_artifacts(req: func.HttpRequest):
    ctx = get_context()
    notebook_id = req.route_params["notebook_id"]
    ctx.store.get_notebook(notebook_id)
    return ctx.store.list_artifacts(notebook_id)


@app.route(route="notebooks/{notebook_id}/artifacts", methods=["POST"])
@api
def create_artifact(req: func.HttpRequest):
    body = _body(req)
    try:
        question_count = int(body.get("question_count") or 10)
    except (TypeError, ValueError) as exc:
        raise ValidationError("question_count must be a number.") from exc
    artifact = artifacts.generate(
        get_context(),
        req.route_params["notebook_id"],
        str(body.get("kind", "")),
        str(body.get("instructions") or ""),
        question_count,
    )
    return _json(artifact, 201)


@app.route(route="notebooks/{notebook_id}/artifacts/{artifact_id}", methods=["GET"])
@api
def get_artifact(req: func.HttpRequest):
    return artifacts.read(get_context(), req.route_params["notebook_id"], req.route_params["artifact_id"])


# --------------------------------------------------------------------- health


@app.route(route="health", methods=["GET"])
@api
def health(req: func.HttpRequest):
    ctx = get_context()  # raises a ConfigError with the missing variable's name if misconfigured
    return {
        "status": "ok",
        "index": ctx.settings.search_index_name,
        "embedding_model": ctx.settings.embedding_model,
        "llm_model": ctx.settings.llm_model,
        "retrieval_mode": ctx.settings.retrieval_mode,
    }
