from __future__ import annotations

from pathlib import Path

from mare.chat import _build_app_from_args, _discover_folder_inputs, run_chat
from mare.types import Modality, QueryPlan, RetrievalExplanation, RetrievalHit


class _FakeApp:
    def __init__(self) -> None:
        self.corpus_path = Path("generated/manual.json")
        self.corpus_paths = [self.corpus_path]
        self.source_pdf = Path("manual.pdf")
        self.source_pdfs = [self.source_pdf]
        self.documents = [object()]

    def describe_corpus(self, page_limit: int = 3, object_limit: int = 5):
        return {"page_count": 1, "object_counts": {"procedure": 1}}

    def search_objects(self, query: str, object_type: str | None = None, limit: int = 5):
        return [{"page": 10, "object_type": object_type or "procedure", "content": "Connect the AC adapter."}]

    def explain(self, query: str, top_k: int = 3):
        return RetrievalExplanation(
            plan=QueryPlan(
                query=query,
                selected_modalities=[Modality.TEXT],
                discarded_modalities=[Modality.IMAGE, Modality.LAYOUT],
                confidence=0.8,
                intent="semantic_lookup",
                rationale="test",
            ),
            per_modality_results={},
            fused_results=[
                RetrievalHit(
                    doc_id="doc-1",
                    title="Manual",
                    page=10,
                    modality=Modality.TEXT,
                    score=0.95,
                    reason="Matched text terms: adapter",
                    snippet="Connect the AC adapter to the laptop.",
                    page_image_path="generated/manual/page-10.png",
                    highlight_image_path="generated/manual/highlight-10.png",
                    object_id="doc-1:procedure:1",
                    object_type="procedure",
                    metadata={"source": "manual.pdf"},
                )
            ],
        )


def test_discover_folder_inputs(tmp_path: Path) -> None:
    (tmp_path / "manual.pdf").write_text("pdf")
    (tmp_path / "manual.json").write_text("{}")

    pdfs, corpora = _discover_folder_inputs(tmp_path)

    assert pdfs == [str(tmp_path / "manual.pdf")]
    assert corpora == [str(tmp_path / "manual.json")]


def test_build_app_from_folder_uses_discovered_inputs(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "manual.pdf").write_text("pdf")
    fake_app = _FakeApp()
    monkeypatch.setattr("mare.chat._load_app", lambda **kwargs: fake_app)

    app = _build_app_from_args(folder=str(tmp_path), pdfs=[], corpora=[], reuse=True, parser="builtin")

    assert app is fake_app


def test_run_chat_answers_question_and_exits(monkeypatch, capsys) -> None:
    answers = iter(["how do I connect the AC adapter", ":quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    run_chat(_FakeApp(), top_k=3, page_limit=3, object_limit=5)
    output = capsys.readouterr().out

    assert "MARE Chat" in output
    assert "Best page: 10" in output
    assert "Highlight:" in output


def test_run_chat_supports_json_and_sources(monkeypatch, capsys) -> None:
    answers = iter([":sources", ":json how do I connect the AC adapter", ":quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    run_chat(_FakeApp(), top_k=3, page_limit=3, object_limit=5)
    output = capsys.readouterr().out

    assert "Sources" in output
    assert "\"workflow\": \"agent-evidence\"" in output
