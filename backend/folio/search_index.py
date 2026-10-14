"""Azure AI Search index: schema, writes, deletes, and the retrieval modes compared in the evaluation."""

from __future__ import annotations

from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from .config import RETRIEVAL_MODES, Settings
from .errors import FolioError, ValidationError
from .storage import check_id, chunked

SEMANTIC_CONFIG = "default"
_VECTOR_PROFILE = "default-profile"
_HNSW = "default-hnsw"
_UPLOAD_BATCH = 100  # keeps each request well under the 16 MB payload limit even with large vectors
_RESULT_FIELDS = [
    "id", "notebook_id", "source_id", "source_type", "source_name", "source_url", "page", "chunk_index", "content",
]


class SearchIndexError(FolioError):
    status_code = 502


class ChunkIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        credential = AzureKeyCredential(settings.search_api_key)
        self._index_client = SearchIndexClient(settings.search_endpoint, credential)
        self._client = SearchClient(settings.search_endpoint, settings.search_index_name, credential)

    # ------------------------------------------------------------------ schema

    def build_index(self) -> SearchIndex:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SimpleField(name="notebook_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="source_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source_url", type=SearchFieldDataType.String),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.settings.embedding_dimensions,
                vector_search_profile_name=_VECTOR_PROFILE,
            ),
        ]
        return SearchIndex(
            name=self.settings.search_index_name,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name=_HNSW)],
                profiles=[VectorSearchProfile(name=_VECTOR_PROFILE, algorithm_configuration_name=_HNSW)],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=SEMANTIC_CONFIG,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="source_name"),
                            content_fields=[SemanticField(field_name="content")],
                        ),
                    )
                ]
            ),
        )

    def ensure_index(self) -> None:
        try:
            existing = self._index_client.get_index(self.settings.search_index_name)
        except ResourceNotFoundError:
            self._index_client.create_index(self.build_index())
            return
        vector_field = next((f for f in existing.fields if f.name == "content_vector"), None)
        dims = getattr(vector_field, "vector_search_dimensions", None)
        if dims and dims != self.settings.embedding_dimensions:
            raise SearchIndexError(
                f"Index '{self.settings.search_index_name}' stores {dims}-dimensional vectors but "
                f"EMBEDDING_DIMENSIONS is {self.settings.embedding_dimensions}. Use a new AZURE_SEARCH_INDEX_NAME."
            )

    # ------------------------------------------------------------------ writes

    def upload(self, documents: list[dict[str, Any]]) -> None:
        for batch in chunked(documents, _UPLOAD_BATCH):
            results = self._client.upload_documents(documents=batch)
            failed = [r for r in results if not r.succeeded]
            if failed:
                raise SearchIndexError(f"Search rejected {len(failed)} chunk(s): {failed[0].error_message}")

    def delete_where(self, odata_filter: str) -> int:
        """Search has no delete-by-query, so collect the matching keys first, then delete them."""
        keys: list[dict[str, str]] = []
        while True:
            page = list(
                self._client.search(
                    search_text="*",
                    filter=odata_filter,
                    select=["id"],
                    order_by=["source_id asc", "chunk_index asc"],
                    top=1000,
                    skip=len(keys),
                )
            )
            keys.extend({"id": r["id"]} for r in page)
            if len(page) < 1000:
                break
        for batch in chunked(keys, 1000):
            self._client.delete_documents(documents=batch)
        return len(keys)

    def delete_notebook(self, notebook_id: str) -> int:
        return self.delete_where(f"notebook_id eq '{check_id(notebook_id)}'")

    def delete_source(self, source_id: str) -> int:
        return self.delete_where(f"source_id eq '{check_id(source_id)}'")

    # ------------------------------------------------------------------- reads

    def retrieve(
        self, notebook_id: str, query: str, query_vector: list[float] | None, mode: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Return the top_k chunks for a notebook using one of the retrieval modes:

        keyword          BM25 full-text only
        vector           embedding similarity only (HNSW)
        hybrid           keyword + vector, fused with Reciprocal Rank Fusion
        hybrid_semantic  hybrid, then re-ranked by Azure's semantic ranker (cross-encoder)
        """
        if mode not in RETRIEVAL_MODES:
            raise ValidationError(f"Unknown retrieval mode '{mode}'.")
        kwargs: dict[str, Any] = {
            "filter": f"notebook_id eq '{check_id(notebook_id)}'",
            "select": _RESULT_FIELDS,
            "top": top_k,
        }
        if mode != "keyword":
            if query_vector is None:
                raise ValidationError("A query vector is required for vector retrieval.")
            # Hybrid modes pull a wider vector candidate set so fusion/re-ranking has more to choose from.
            k = top_k if mode == "vector" else max(top_k * 5, 25)
            kwargs["vector_queries"] = [VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="content_vector")]
        if mode == "hybrid_semantic":
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG

        search_text = None if mode == "vector" else query
        try:
            results = list(self._client.search(search_text=search_text, **kwargs))
        except HttpResponseError as exc:
            hint = " Is the semantic ranker enabled on the Search service?" if mode == "hybrid_semantic" else ""
            raise SearchIndexError(f"Search query failed: {exc.message}{hint}") from exc

        return [
            {
                **{field: r.get(field) for field in _RESULT_FIELDS},
                "score": r.get("@search.score"),
                "reranker_score": r.get("@search.reranker_score"),
            }
            for r in results
        ]

    def notebook_chunks(self, notebook_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        """All chunks of a notebook in document order (used for reports and quizzes)."""
        results = self._client.search(
            search_text="*",
            filter=f"notebook_id eq '{check_id(notebook_id)}'",
            select=_RESULT_FIELDS,
            order_by=["source_id asc", "chunk_index asc"],
            top=limit,
        )
        return [{field: r.get(field) for field in _RESULT_FIELDS} for r in results]
