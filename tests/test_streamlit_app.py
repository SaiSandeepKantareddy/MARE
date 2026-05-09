import json
from types import SimpleNamespace
from pathlib import Path

from mare.streamlit_app import (
    _append_ui_session_history,
    _build_run_signature,
    _clear_ui_session_history,
    _load_ui_session_history,
    _result_matches_signature,
)


def _stack_controls(**overrides):
    controls = {
        "mode": "Advanced",
        "parser": {"value": "builtin"},
        "retriever": {"value": "builtin"},
        "reranker": {"value": "none"},
        "output": {"value": "mare"},
        "reuse_corpus": False,
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "mare-docs",
        "qdrant_index_before_query": False,
    }
    controls.update(overrides)
    return controls


def test_run_signature_changes_when_stack_changes() -> None:
    baseline = _build_run_signature(["manual.pdf"], "how do I connect the AC adapter", 3, _stack_controls())
    changed = _build_run_signature(
        ["manual.pdf"],
        "how do I connect the AC adapter",
        3,
        _stack_controls(retriever={"value": "sentence-transformers"}),
    )

    assert baseline != changed


def test_result_matches_signature_only_for_current_inputs() -> None:
    signature = _build_run_signature(["manual.pdf"], "configure wake on lan", 2, _stack_controls())
    result = {"run_signature": signature}

    assert _result_matches_signature(result, signature) is True

    different_signature = _build_run_signature(["manual.pdf"], "configure wake on lan", 4, _stack_controls())
    assert _result_matches_signature(result, different_signature) is False


def test_run_signature_changes_when_uploaded_file_set_changes() -> None:
    baseline = _build_run_signature(["manual-a.pdf"], "where is wake on lan discussed", 3, _stack_controls())
    changed = _build_run_signature(["manual-a.pdf", "manual-b.pdf"], "where is wake on lan discussed", 3, _stack_controls())

    assert baseline != changed


def test_ui_session_history_append_persists_recent_run(tmp_path: Path) -> None:
    history_path = tmp_path / "ui-history.json"
    explanation = SimpleNamespace(
        plan=SimpleNamespace(intent="semantic_lookup"),
        fused_results=[
            SimpleNamespace(
                title="Manual",
                page=10,
                metadata={"source": "manual.pdf"},
                object_type="procedure",
                snippet="Connect the AC adapter to the laptop.",
            )
        ],
    )
    stack = {
        "parser": "Builtin PDF",
        "retriever": "Built-in lexical (Recommended)",
        "reranker": "None",
        "output_mode": "MARE evidence",
    }

    history = _append_ui_session_history(
        filenames=["manual.pdf"],
        query="how do I connect the AC adapter",
        explanation=explanation,
        stack=stack,
        path=history_path,
    )

    payload = json.loads(history_path.read_text())
    assert len(history["entries"]) == 1
    assert payload["entries"][0]["query"] == "how do I connect the AC adapter"
    assert payload["entries"][0]["citation"] == "manual.pdf | page 10"
    assert payload["entries"][0]["stack"]["parser"] == "Builtin PDF"


def test_ui_session_history_clear_removes_entries(tmp_path: Path) -> None:
    history_path = tmp_path / "ui-history.json"
    history_path.write_text(json.dumps({"created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00", "entries": [{"query": "test"}]}))

    history = _clear_ui_session_history(history_path)
    reloaded = _load_ui_session_history(history_path)

    assert history["entries"] == []
    assert reloaded["entries"] == []
