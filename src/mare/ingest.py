from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mare.types import Document


def _require_pypdf():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF ingestion. Install dependencies with `pip install -e .` "
            "or `pip install pypdf`."
        ) from exc
    return PdfReader


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _infer_layout_hints(text: str) -> str:
    hints: list[str] = []
    lowered = text.lower()

    if "table" in lowered:
        hints.append("table")
    if "figure" in lowered or "fig." in lowered:
        hints.append("figure")
    if "abstract" in lowered:
        hints.append("abstract")
    if "references" in lowered:
        hints.append("references")

    return " ".join(hints)


def ingest_pdf(pdf_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    PdfReader = _require_pypdf()
    pdf_file = Path(pdf_path)
    reader = PdfReader(str(pdf_file))
    title = pdf_file.stem
    documents: list[Document] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = _normalize_text(raw_text)
        if not text:
            text = f"[No extractable text found on page {idx}]"

        documents.append(
            Document(
                doc_id=f"{pdf_file.stem.lower().replace(' ', '-')}-p{idx}",
                title=title,
                page=idx,
                text=text,
                image_caption="",
                layout_hints=_infer_layout_hints(text),
                metadata={
                    "source": str(pdf_file),
                    "collection": "pdf-ingest",
                },
            )
        )

    payload = {
        "source_pdf": str(pdf_file),
        "documents": [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "page": doc.page,
                "text": doc.text,
                "image_caption": doc.image_caption,
                "layout_hints": doc.layout_hints,
                "metadata": doc.metadata,
            }
            for doc in documents
        ],
    }

    if output_path is not None:
        Path(output_path).write_text(json.dumps(payload, indent=2))

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into a MARE JSON corpus")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path. Defaults to generated/<pdf-stem>.json",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output = Path(args.output) if args.output else Path("generated") / f"{pdf_path.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ingest_pdf(pdf_path=pdf_path, output_path=output)

    print(
        json.dumps(
            {
                "source_pdf": payload["source_pdf"],
                "pages_indexed": len(payload["documents"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
