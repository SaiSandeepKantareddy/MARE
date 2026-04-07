from pathlib import Path

import mare.ingest as ingest_module


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, _path: str) -> None:
        self.pages = [
            _FakePage("Abstract Figure 1 introduces the system."),
            _FakePage("Table 2 compares the models."),
        ]


def test_ingest_pdf_writes_page_level_corpus(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ingest_module, "_require_pypdf", lambda: _FakeReader)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder")
    output_path = tmp_path / "sample.json"

    payload = ingest_module.ingest_pdf(pdf_path=pdf_path, output_path=output_path)

    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["page"] == 1
    assert payload["documents"][0]["layout_hints"] == "figure abstract"
    assert payload["documents"][1]["layout_hints"] == "table"
    assert output_path.exists()
