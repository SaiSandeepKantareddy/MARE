from __future__ import annotations

"""
Example: retrieve grounded evidence across multiple MARE corpora.

Typical usage:

PYTHONPATH=src python3 examples/multi_pdf_workflow.py \
  --corpus generated/manual-a.json \
  --corpus generated/manual-b.json \
  --query "where is wake on lan discussed"
"""

import argparse
import json

from mare import load_corpora


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MARE retrieval across multiple corpus JSON files")
    parser.add_argument("--corpus", dest="corpora", action="append", required=True, help="Path to a MARE corpus JSON")
    parser.add_argument("--query", required=True, help="Question to ask across the corpus set")
    parser.add_argument("--top-k", type=int, default=5, help="How many results to return")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    app = load_corpora(args.corpora)
    explanation = app.explain(args.query, top_k=args.top_k)

    print(
        json.dumps(
            {
                "workflow": "multi-pdf-evidence",
                "query": args.query,
                "source": {
                    "corpus_count": len(app.corpus_paths),
                    "corpus_paths": [str(path) for path in app.corpus_paths],
                    "source_pdfs": [str(path) for path in app.source_pdfs],
                    "documents": len(app.documents),
                },
                "plan": {
                    "intent": explanation.plan.intent,
                    "selected_modalities": [item.value for item in explanation.plan.selected_modalities],
                    "discarded_modalities": [item.value for item in explanation.plan.discarded_modalities],
                    "confidence": explanation.plan.confidence,
                    "rationale": explanation.plan.rationale,
                },
                "results": [
                    {
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
                        "metadata": hit.metadata,
                    }
                    for hit in explanation.fused_results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
