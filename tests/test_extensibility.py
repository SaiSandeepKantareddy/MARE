from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import mare.ingest as ingest_module
from mare import FastEmbedReranker, IdentityReranker, KeywordBoostReranker, MAREApp, QdrantHybridRetriever
from mare.extensions import DoclingParser, MAREConfig, UnstructuredParser, get_parser
from mare.retrievers.base import BaseRetriever
from mare.types import Document, Modality, RetrievalHit


class _CustomParser:
    def ingest(self, pdf_path: Path, output_path: Path) -> Path:
        payload = {
            "source_pdf": str(pdf_path),
            "documents": [
                {
                    "doc_id": "custom-p1",
                    "title": "Custom",
                    "page": 1,
                    "text": "Custom parser output for setup instructions.",
                    "image_caption": "",
                    "layout_hints": "",
                    "page_image_path": "",
                    "objects": [],
                    "metadata": {"source": str(pdf_path)},
                }
            ],
        }
        output_path.write_text(json.dumps(payload))
        return output_path


class _AlwaysTopTextRetriever(BaseRetriever):
    modality = Modality.TEXT

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                doc_id="override-1",
                title="Override",
                page=9,
                modality=self.modality,
                score=0.99,
                reason=f"Custom retriever handled: {query}",
                snippet="Overridden retrieval result.",
            )
        ][:top_k]


class _ReverseReranker:
    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 5) -> list[RetrievalHit]:
        return list(reversed(hits))[:top_k]


def test_custom_parser_can_build_a_corpus(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder")
    output_path = tmp_path / "sample.json"

    app = MAREApp.from_pdf(pdf_path=pdf_path, output_path=output_path, parser=_CustomParser())

    assert app.documents[0].doc_id == "custom-p1"
    assert app.documents[0].text.startswith("Custom parser output")


def test_custom_retriever_factory_can_override_default_retrieval() -> None:
    docs = [Document(doc_id="base-1", title="Base", page=1, text="Base document text.")]
    config = MAREConfig(retriever_factories={Modality.TEXT: lambda documents: _AlwaysTopTextRetriever(documents)})

    app = MAREApp.from_documents(docs, config=config)
    best = app.best_match("anything")

    assert best is not None
    assert best.doc_id == "override-1"
    assert "Custom retriever handled" in best.reason


def test_custom_reranker_can_reorder_fused_results() -> None:
    docs = [
        Document(doc_id="1", title="Doc1", page=1, text="adapter instructions and setup"),
        Document(doc_id="2", title="Doc2", page=2, text="adapter instructions and setup"),
    ]
    config = MAREConfig(reranker=_ReverseReranker())

    app = MAREApp.from_documents(docs, config=config)
    results = app.retrieve("adapter instructions", top_k=2)

    assert len(results) == 2
    assert results[0].doc_id == "2"


def test_builtin_parsers_are_discoverable() -> None:
    assert get_parser("builtin") is not None
    assert get_parser("docling") is not None
    assert get_parser("unstructured") is not None


def test_identity_reranker_preserves_order() -> None:
    hits = [
        RetrievalHit(doc_id="1", title="A", page=1, modality=Modality.TEXT, score=0.9, reason="a"),
        RetrievalHit(doc_id="2", title="B", page=2, modality=Modality.TEXT, score=0.8, reason="b"),
    ]
    reranked = IdentityReranker().rerank("adapter", hits, top_k=2)

    assert [hit.doc_id for hit in reranked] == ["1", "2"]


def test_keyword_boost_reranker_prefers_label_overlap() -> None:
    hits = [
        RetrievalHit(
            doc_id="1",
            title="Table",
            page=1,
            modality=Modality.TEXT,
            score=0.6,
            reason="table result",
            snippet="comparison baseline",
            metadata={"label": "Table 2"},
        ),
        RetrievalHit(
            doc_id="2",
            title="Section",
            page=2,
            modality=Modality.TEXT,
            score=0.6,
            reason="plain result",
            snippet="plain text",
        ),
    ]
    reranked = KeywordBoostReranker().rerank("comparison table", hits, top_k=2)

    assert reranked[0].doc_id == "1"


def test_unstructured_parser_builds_mare_corpus_with_fake_module(tmp_path: Path, monkeypatch) -> None:
    class _FakeMetadata:
        def __init__(self, page_number: int) -> None:
            self.page_number = page_number

    class _FakeElement:
        def __init__(self, text: str, category: str, page_number: int) -> None:
            self.text = text
            self.category = category
            self.metadata = _FakeMetadata(page_number)

    def _fake_partition_pdf(filename: str, strategy: str, include_page_breaks: bool):
        assert strategy == "hi_res"
        assert include_page_breaks is True
        return [
            _FakeElement("Wake on LAN feature", "Title", 1),
            _FakeElement("Table 1. Settings matrix", "Table", 1),
            _FakeElement("Architecture diagram", "Image", 2),
        ]

    fake_pdf_module = types.ModuleType("unstructured.partition.pdf")
    fake_pdf_module.partition_pdf = _fake_partition_pdf
    monkeypatch.setitem(sys.modules, "unstructured", types.ModuleType("unstructured"))
    monkeypatch.setitem(sys.modules, "unstructured.partition", types.ModuleType("unstructured.partition"))
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", fake_pdf_module)
    monkeypatch.setattr(
        ingest_module,
        "_render_page_images",
        lambda pdf_path, image_dir, scale=1.5: [str(image_dir / "page-1.png"), str(image_dir / "page-2.png")],
    )

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder")
    output_path = tmp_path / "sample.json"

    parser = UnstructuredParser()
    parser.ingest(pdf_path, output_path)
    payload = json.loads(output_path.read_text())

    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["metadata"]["parser"] == "unstructured"
    assert any(obj["object_type"] == "table" for obj in payload["documents"][0]["objects"])
    assert any(obj["object_type"] == "figure" for obj in payload["documents"][1]["objects"])


def test_docling_parser_builds_mare_corpus_with_fake_module(tmp_path: Path, monkeypatch) -> None:
    class _FakePage:
        def __init__(self, page_no: int, assembled: str) -> None:
            self.page_no = page_no
            self.assembled = assembled

    class _FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Fallback markdown"

    class _FakeResult:
        def __init__(self) -> None:
            self.pages = [
                _FakePage(1, "Wake on LAN feature"),
                _FakePage(2, "Table 1. Settings matrix"),
            ]
            self.document = _FakeDocument()
            self.confidence = 0.91

    class _FakeDocumentConverter:
        def convert(self, source: str):
            assert source.endswith("sample.pdf")
            return _FakeResult()

    fake_docling_converter = types.ModuleType("docling.document_converter")
    fake_docling_converter.DocumentConverter = _FakeDocumentConverter
    monkeypatch.setitem(sys.modules, "docling", types.ModuleType("docling"))
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_docling_converter)
    monkeypatch.setattr(
        ingest_module,
        "_render_page_images",
        lambda pdf_path, image_dir, scale=1.5: [str(image_dir / "page-1.png"), str(image_dir / "page-2.png")],
    )

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder")
    output_path = tmp_path / "sample.json"

    parser = DoclingParser()
    parser.ingest(pdf_path, output_path)
    payload = json.loads(output_path.read_text())

    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["metadata"]["parser"] == "docling"
    assert payload["documents"][0]["metadata"]["confidence"] == "0.91"
    assert payload["documents"][1]["text"].startswith("Table 1")


def test_fastembed_reranker_uses_cross_encoder_scores(monkeypatch) -> None:
    class _FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def rerank(self, query: str, documents: list[str]):
            assert query == "comparison table"
            return [0.2, 0.9]

    fake_cross_encoder_module = types.ModuleType("fastembed.rerank.cross_encoder")
    fake_cross_encoder_module.TextCrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "fastembed", types.ModuleType("fastembed"))
    monkeypatch.setitem(sys.modules, "fastembed.rerank", types.ModuleType("fastembed.rerank"))
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", fake_cross_encoder_module)

    hits = [
        RetrievalHit(doc_id="1", title="A", page=1, modality=Modality.TEXT, score=0.1, reason="a", snippet="alpha"),
        RetrievalHit(doc_id="2", title="B", page=2, modality=Modality.TEXT, score=0.1, reason="b", snippet="beta"),
    ]

    reranked = FastEmbedReranker().rerank("comparison table", hits, top_k=2)

    assert [hit.doc_id for hit in reranked] == ["2", "1"]
    assert reranked[0].score == 0.9


def test_qdrant_hybrid_retriever_maps_payloads_to_mare_hits(monkeypatch) -> None:
    class _FakeDocument:
        def __init__(self, text: str, model: str) -> None:
            self.text = text
            self.model = model

    class _FakePoint:
        def __init__(self, point_id: str, score: float, payload: dict) -> None:
            self.id = point_id
            self.score = score
            self.payload = payload

    class _FakeResponse:
        def __init__(self, points) -> None:
            self.points = points

    class _FakeClient:
        def query_points(self, **kwargs):
            query = kwargs["query"]
            assert query.text == "wake on lan"
            assert kwargs["collection_name"] == "mare-docs"
            assert kwargs["using"] == "text"
            return _FakeResponse(
                [
                    _FakePoint(
                        "doc-61",
                        0.83,
                        {
                            "doc_id": "doc-61",
                            "title": "Manual",
                            "page": 61,
                            "snippet": "Wake on LAN feature...",
                            "page_image_path": "generated/manual/page-61.png",
                            "object_type": "procedure",
                            "metadata": {"label": "Wake on LAN"},
                        },
                    )
                ]
            )

    fake_models = types.SimpleNamespace(Document=_FakeDocument)
    fake_qdrant = types.ModuleType("qdrant_client")
    fake_qdrant.models = fake_models
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant)

    retriever = QdrantHybridRetriever(
        [],
        collection_name="mare-docs",
        client=_FakeClient(),
        vector_name="text",
    )
    hits = retriever.retrieve("wake on lan", top_k=1)

    assert len(hits) == 1
    assert hits[0].doc_id == "doc-61"
    assert hits[0].object_type == "procedure"
    assert hits[0].metadata["label"] == "Wake on LAN"
