from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from mare.ingest import ingest_pdf
from mare.objects import extract_document_objects
from mare.retrievers.base import BaseRetriever
from mare.types import DocumentObject, Modality, ObjectType, RetrievalHit


class DocumentParser(Protocol):
    """Build or update a MARE corpus from a source document."""

    def ingest(self, pdf_path: Path, output_path: Path) -> Path:
        """Return the path to the generated corpus file."""


class ResultReranker(Protocol):
    """Optional second-stage reranker for fused results."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 5) -> list[RetrievalHit]:
        """Return reranked hits, usually on a small candidate set."""


RetrieverFactory = Callable[[list], BaseRetriever]


@dataclass
class MAREConfig:
    parser: str | DocumentParser | None = None
    retriever_factories: dict[Modality, RetrieverFactory] = field(default_factory=dict)
    reranker: ResultReranker | None = None


class BuiltinPDFParser:
    """Default local parser that uses MARE's built-in PDF ingestion."""

    def ingest(self, pdf_path: Path, output_path: Path) -> Path:
        ingest_pdf(pdf_path=pdf_path, output_path=output_path)
        return output_path


class DoclingParser:
    """Placeholder adapter for Docling-backed parsing.

    This keeps MARE's parser interface stable while letting developers wire in
    richer OCR/layout/table extraction when Docling is available in their env.
    """

    def ingest(self, pdf_path: Path, output_path: Path) -> Path:
        raise RuntimeError(
            "DoclingParser is an integration stub. Install/configure Docling in your environment and "
            "implement this adapter to emit a MARE-compatible corpus."
        )


class UnstructuredParser:
    """Parser adapter backed by Unstructured's PDF partitioning pipeline."""

    def __init__(self, strategy: str = "hi_res") -> None:
        self.strategy = strategy

    def ingest(self, pdf_path: Path, output_path: Path) -> Path:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise RuntimeError(
                "UnstructuredParser requires `unstructured[pdf]`. Install it with "
                "`pip install 'mare-retrieval[unstructured]'` or `pip install 'unstructured[pdf]'`."
            ) from exc

        from mare.ingest import _infer_layout_hints, _infer_page_signals, _normalize_text, _render_page_images

        output_path.parent.mkdir(parents=True, exist_ok=True)
        page_images = _render_page_images(pdf_path, output_path.with_suffix(""))
        elements = partition_pdf(filename=str(pdf_path), strategy=self.strategy, include_page_breaks=True)

        pages: dict[int, dict[str, object]] = {}

        for element in elements:
            metadata = getattr(element, "metadata", None)
            if metadata is None:
                page_number = 1
            elif isinstance(metadata, dict):
                page_number = int(metadata.get("page_number", 1) or 1)
            else:
                page_number = int(getattr(metadata, "page_number", 1) or 1)

            page_entry = pages.setdefault(page_number, {"lines": [], "objects": []})
            text = _normalize_text(getattr(element, "text", "") or "")
            if text:
                page_entry["lines"].append(text)

            category = getattr(element, "category", element.__class__.__name__)
            object_type = self._map_category_to_object_type(category)
            if object_type is None or not text:
                continue

            page_entry["objects"].append(
                DocumentObject(
                    object_id=f"{pdf_path.stem.lower().replace(' ', '-')}-{page_number}:{object_type.value}:{len(page_entry['objects']) + 1}",
                    doc_id=f"{pdf_path.stem.lower().replace(' ', '-')}-p{page_number}",
                    page=page_number,
                    object_type=object_type,
                    content=text,
                    metadata={"label": category, "source": "unstructured"},
                )
            )

        payload_documents = []
        max_page = max(len(page_images), max(pages.keys(), default=0))
        for page_number in range(1, max_page + 1):
            page_entry = pages.get(page_number, {"lines": [], "objects": []})
            raw_text = "\n".join(page_entry["lines"]).strip()
            text = _normalize_text(raw_text) if raw_text else f"[No extractable text found on page {page_number}]"
            doc_id = f"{pdf_path.stem.lower().replace(' ', '-')}-p{page_number}"
            objects = extract_document_objects(raw_text or text, doc_id, page_number)
            objects.extend(page_entry["objects"])

            payload_documents.append(
                {
                    "doc_id": doc_id,
                    "title": pdf_path.stem,
                    "page": page_number,
                    "text": text,
                    "image_caption": "",
                    "layout_hints": _infer_layout_hints(text),
                    "page_image_path": page_images[page_number - 1] if page_number - 1 < len(page_images) else "",
                    "objects": [
                        {
                            "object_id": obj.object_id,
                            "doc_id": obj.doc_id,
                            "page": obj.page,
                            "object_type": obj.object_type.value,
                            "content": obj.content,
                            "metadata": obj.metadata,
                        }
                        for obj in objects
                    ],
                    "metadata": {
                        "source": str(pdf_path),
                        "collection": "unstructured-ingest",
                        "signals": _infer_page_signals(text),
                        "parser": "unstructured",
                    },
                }
            )

        output_path.write_text(json.dumps({"source_pdf": str(pdf_path), "documents": payload_documents}, indent=2))
        return output_path

    @staticmethod
    def _map_category_to_object_type(category: str) -> ObjectType | None:
        normalized = category.lower()
        if "table" in normalized:
            return ObjectType.TABLE
        if "image" in normalized or "figure" in normalized or "picture" in normalized:
            return ObjectType.FIGURE
        if "title" in normalized or "header" in normalized:
            return ObjectType.SECTION
        return None


class IdentityReranker:
    """No-op reranker useful as a baseline or composition default."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 5) -> list[RetrievalHit]:
        return hits[:top_k]


class KeywordBoostReranker:
    """Small built-in reranker that rewards exact term overlap and metadata labels."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 5) -> list[RetrievalHit]:
        query_terms = set(query.lower().split())
        rescored: list[tuple[float, RetrievalHit]] = []
        for hit in hits:
            label_terms = set(str(hit.metadata.get("label", "")).lower().split())
            text_terms = set((hit.snippet or hit.reason or "").lower().split())
            overlap = len(query_terms & (label_terms | text_terms))
            rescored.append((hit.score + (0.03 * overlap), hit))

        rescored.sort(key=lambda item: item[0], reverse=True)
        reranked: list[RetrievalHit] = []
        for score, hit in rescored[:top_k]:
            hit.score = round(score, 4)
            reranked.append(hit)
        return reranked


class FastEmbedReranker:
    """Reranker backed by FastEmbed cross-encoders.

    Based on Qdrant/FastEmbed's documented `TextCrossEncoder` usage.
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "FastEmbedReranker requires FastEmbed. Install it with "
                    "`pip install 'mare-retrieval[fastembed]'` or `pip install fastembed`."
                ) from exc
            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 5) -> list[RetrievalHit]:
        if not hits:
            return []

        model = self._get_model()
        documents = [hit.snippet or hit.reason or hit.title for hit in hits]
        scores = list(model.rerank(query, documents))

        rescored: list[tuple[float, RetrievalHit]] = []
        for hit, rerank_score in zip(hits, scores):
            combined_score = float(rerank_score)
            hit.score = round(combined_score, 4)
            rescored.append((combined_score, hit))

        rescored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in rescored[:top_k]]


class QdrantHybridRetriever(BaseRetriever):
    """Retriever backed by Qdrant query APIs.

    This is intended for developers who already have a Qdrant collection with
    payload fields compatible with MARE's result shape.
    """

    modality = Modality.TEXT

    def __init__(
        self,
        documents: list,
        *,
        collection_name: str,
        client=None,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        vector_name: str | None = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        payload_text_key: str = "text",
    ) -> None:
        super().__init__(documents)
        self.collection_name = collection_name
        self.client = client
        self.url = url
        self.api_key = api_key
        self.location = location
        self.vector_name = vector_name
        self.embedding_model = embedding_model
        self.payload_text_key = payload_text_key

    def _get_client(self):
        if self.client is not None:
            return self.client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "QdrantHybridRetriever requires `qdrant-client`. Install it with "
                "`pip install 'mare-retrieval[integrations]'` or `pip install qdrant-client[fastembed]`."
            ) from exc
        self.client = QdrantClient(url=self.url, api_key=self.api_key, location=self.location)
        return self.client

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        client = self._get_client()
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "QdrantHybridRetriever requires `qdrant-client`. Install it with "
                "`pip install 'mare-retrieval[integrations]'` or `pip install qdrant-client[fastembed]`."
            ) from exc

        query_document = models.Document(text=query, model=self.embedding_model)
        query_kwargs = {
            "collection_name": self.collection_name,
            "query": query_document,
            "with_payload": True,
            "limit": top_k,
        }
        if self.vector_name:
            query_kwargs["using"] = self.vector_name

        response = client.query_points(**query_kwargs)
        points = getattr(response, "points", response)

        hits: list[RetrievalHit] = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            snippet = str(payload.get("snippet") or payload.get(self.payload_text_key) or "")
            title = str(payload.get("title") or payload.get("doc_id") or "Qdrant result")
            metadata = dict(payload.get("metadata") or {})
            if payload.get("label") and "label" not in metadata:
                metadata["label"] = str(payload["label"])

            hits.append(
                RetrievalHit(
                    doc_id=str(payload.get("doc_id") or getattr(point, "id", "")),
                    title=title,
                    page=int(payload.get("page") or 0),
                    modality=self.modality,
                    score=round(float(getattr(point, "score", 0.0)), 4),
                    reason=str(payload.get("reason") or f"Qdrant hit from collection '{self.collection_name}'"),
                    snippet=snippet,
                    page_image_path=str(payload.get("page_image_path") or ""),
                    highlight_image_path=str(payload.get("highlight_image_path") or ""),
                    object_id=str(payload.get("object_id") or ""),
                    object_type=str(payload.get("object_type") or ""),
                    metadata=metadata,
                )
            )

        return hits


_PARSER_REGISTRY: dict[str, DocumentParser] = {
    "builtin": BuiltinPDFParser(),
    "docling": DoclingParser(),
    "unstructured": UnstructuredParser(),
}


def register_parser(name: str, parser: DocumentParser) -> None:
    _PARSER_REGISTRY[name] = parser


def get_parser(name: str) -> DocumentParser:
    try:
        return _PARSER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PARSER_REGISTRY))
        raise KeyError(f"Unknown parser '{name}'. Available parsers: {available}") from exc
