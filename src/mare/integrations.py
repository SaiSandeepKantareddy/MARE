from __future__ import annotations

from pathlib import Path
from typing import Any

from mare.types import RetrievalHit


def format_evidence_citation(
    *,
    title: str,
    page: int,
    metadata: dict[str, Any] | None = None,
) -> str:
    resolved_metadata = metadata or {}
    source = str(resolved_metadata.get("source") or "").strip()
    source_label = Path(source).name if source else title
    parts = [source_label]

    line_start = str(resolved_metadata.get("line_start") or "").strip()
    line_end = str(resolved_metadata.get("line_end") or "").strip()
    heading = str(resolved_metadata.get("heading") or resolved_metadata.get("label") or "").strip()

    if line_start and line_end:
        if line_start == line_end:
            parts.append(f"line {line_start}")
        else:
            parts.append(f"lines {line_start}-{line_end}")
    else:
        parts.append(f"page {page}")

    if heading:
        parts.append(heading)

    return " | ".join(part for part in parts if part)


def _hit_metadata(hit: RetrievalHit) -> dict[str, Any]:
    metadata = dict(hit.metadata)
    metadata.update(
        {
            "doc_id": hit.doc_id,
            "title": hit.title,
            "page": hit.page,
            "score": hit.score,
            "reason": hit.reason,
            "modality": hit.modality.value,
            "page_image_path": hit.page_image_path,
            "highlight_image_path": hit.highlight_image_path,
            "object_id": hit.object_id,
            "object_type": hit.object_type,
            "citation": format_evidence_citation(title=hit.title, page=hit.page, metadata=hit.metadata),
        }
    )
    return metadata


def _hit_text(hit: RetrievalHit) -> str:
    return hit.snippet or hit.reason or hit.title


def _serialize_hit(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "doc_id": hit.doc_id,
        "title": hit.title,
        "page": hit.page,
        "score": hit.score,
        "snippet": hit.snippet,
        "reason": hit.reason,
        "page_image_path": hit.page_image_path,
        "highlight_image_path": hit.highlight_image_path,
        "object_id": hit.object_id,
        "object_type": hit.object_type,
        "metadata": dict(hit.metadata),
        "citation": format_evidence_citation(title=hit.title, page=hit.page, metadata=hit.metadata),
    }


def _build_comparison_payload(results: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for hit in results:
        metadata = hit.get("metadata", {})
        source_document = str(metadata.get("source") or hit.get("title") or "")
        key = (source_document, str(hit.get("object_type") or "page"), int(hit.get("page") or 0))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        comparison.append(
            {
                "source_document": source_document,
                "citation": hit.get("citation") or "",
                "object_type": hit.get("object_type") or "page",
                "page": hit.get("page"),
                "score": hit.get("score"),
                "reason": hit.get("reason") or "",
                "snippet": hit.get("snippet") or "",
            }
        )
        if len(comparison) >= limit:
            break
    return comparison


def build_grounded_summary_payload(results: list[dict[str, Any]], *, limit: int = 3) -> dict[str, Any]:
    highlights: list[dict[str, Any]] = []
    unique_sources: set[str] = set()
    for hit in results:
        metadata = hit.get("metadata", {})
        source_document = str(metadata.get("source") or hit.get("title") or "")
        if source_document:
            unique_sources.add(source_document)
        highlights.append(
            {
                "citation": hit.get("citation") or "",
                "source_document": source_document,
                "object_type": hit.get("object_type") or "page",
                "snippet": hit.get("snippet") or "",
                "reason": hit.get("reason") or "",
            }
        )
        if len(highlights) >= limit:
            break

    overview = "No grounded evidence found."
    if results:
        result_label = "result" if len(results) == 1 else "results"
        source_count = len(unique_sources)
        source_label = "source" if source_count == 1 else "sources"
        overview = f"Found {len(results)} grounded {result_label} across {source_count} {source_label}."

    return {
        "overview": overview,
        "highlight_count": len(highlights),
        "highlights": highlights,
    }


def hits_to_evidence_payload(query: str, hits: list[RetrievalHit]) -> dict[str, Any]:
    results = [_serialize_hit(hit) for hit in hits]
    return {
        "query": query,
        "results": results,
        "comparison": _build_comparison_payload(results),
        "summary": build_grounded_summary_payload(results),
    }


def hit_to_langchain_document(hit: RetrievalHit):
    try:
        from langchain_core.documents import Document as LangChainDocument
    except ImportError as exc:
        raise RuntimeError(
            "LangChain integration requires `langchain-core`. Install it with "
            "`pip install 'mare-retrieval[langchain]'` or `pip install langchain-core`."
        ) from exc

    return LangChainDocument(page_content=_hit_text(hit), metadata=_hit_metadata(hit))


def hit_to_llamaindex_node(hit: RetrievalHit):
    try:
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex integration requires `llama-index-core`. Install it with "
            "`pip install 'mare-retrieval[llamaindex]'` or `pip install llama-index-core`."
        ) from exc

    node = TextNode(text=_hit_text(hit), metadata=_hit_metadata(hit))
    return NodeWithScore(node=node, score=hit.score)


def create_langchain_retriever(app, top_k: int = 3):
    try:
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise RuntimeError(
            "LangChain integration requires `langchain-core`. Install it with "
            "`pip install 'mare-retrieval[langchain]'` or `pip install langchain-core`."
        ) from exc

    try:
        from pydantic import ConfigDict
    except ImportError:  # pragma: no cover - optional dependency
        ConfigDict = None

    class LangChainMARERetriever(BaseRetriever):
        mare_app: Any
        top_k: int = 3

        if ConfigDict is not None:
            model_config = ConfigDict(arbitrary_types_allowed=True)
        else:  # pragma: no cover - compatibility shim
            class Config:
                arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            hits = self.mare_app.retrieve(query=query, top_k=self.top_k)
            return [hit_to_langchain_document(hit) for hit in hits]

        async def _aget_relevant_documents(self, query: str, *, run_manager=None):
            return self._get_relevant_documents(query, run_manager=run_manager)

    return LangChainMARERetriever(mare_app=app, top_k=top_k)


def create_langgraph_tool(app, top_k: int = 3, name: str = "mare_retrieve", description: str | None = None):
    return create_langchain_tool(app, top_k=top_k, name=name, description=description)


def create_langchain_tool(app, top_k: int = 3, name: str = "mare_retrieve", description: str | None = None):
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError("LangChain tool integration requires `langchain-core`. Install it with "
                           "`pip install 'mare-retrieval[langchain]'` or `pip install langchain-core`.") from exc

    tool_description = description or (
        "Retrieve evidence from documents with MARE. Returns structured page, snippet, "
        "highlight, and metadata for the most relevant results."
    )

    def _run(query: str) -> dict[str, Any]:
        hits = app.retrieve(query=query, top_k=top_k)
        return hits_to_evidence_payload(query, hits)

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=tool_description,
    )


def create_llamaindex_tool(app, top_k: int = 3, name: str = "mare_retrieve", description: str | None = None):
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex tool integration requires `llama-index-core`. Install it with "
            "`pip install 'mare-retrieval[llamaindex]'` or `pip install llama-index-core`."
        ) from exc

    tool_description = description or (
        "Retrieve grounded evidence from documents with MARE and return a structured evidence payload."
    )

    def _run(query: str) -> dict[str, Any]:
        hits = app.retrieve(query=query, top_k=top_k)
        return hits_to_evidence_payload(query, hits)

    return FunctionTool.from_defaults(fn=_run, name=name, description=tool_description)


def create_llamaindex_retriever(app, top_k: int = 3):
    try:
        from llama_index.core.base.base_retriever import BaseRetriever
        from llama_index.core.schema import QueryBundle
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex integration requires `llama-index-core`. Install it with "
            "`pip install 'mare-retrieval[llamaindex]'` or `pip install llama-index-core`."
        ) from exc

    class LlamaIndexMARERetriever(BaseRetriever):
        def __init__(self, mare_app, top_k: int = 3):
            super().__init__()
            self.mare_app = mare_app
            self.top_k = top_k

        def _retrieve(self, query_bundle):
            if isinstance(query_bundle, QueryBundle):
                query = query_bundle.query_str
            else:
                query = getattr(query_bundle, "query_str", str(query_bundle))
            hits = self.mare_app.retrieve(query=query, top_k=self.top_k)
            return [hit_to_llamaindex_node(hit) for hit in hits]

    return LlamaIndexMARERetriever(app, top_k=top_k)
