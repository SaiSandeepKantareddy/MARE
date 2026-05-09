from __future__ import annotations

"""
Example: run MARE over a mixed-document folder and return a structured workflow payload.

Typical usage:

PYTHONPATH=src python3 examples/mixed_docs_workflow.py \
  --folder examples/mixed_docs \
  --query "how do I connect the AC adapter"

Optional folder scoping:

PYTHONPATH=src python3 examples/mixed_docs_workflow.py \
  --folder examples/mixed_docs \
  --include "*.md" \
  --include "*.docx" \
  --query "show me the onboarding steps"
"""

import argparse
import json
from pathlib import Path

from mare.chat import _discover_folder_inputs
from mare.workflow import _build_workflow_payload, _load_app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a MARE workflow over a mixed-document folder")
    parser.add_argument("--folder", required=True, help="Folder containing documents and/or MARE corpora")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional glob pattern relative to --folder to include. Repeat to add more patterns.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Optional glob pattern relative to --folder to exclude. Repeat to add more patterns.",
    )
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="How many retrieval hits to return")
    parser.add_argument("--page-limit", type=int, default=3, help="How many pages to include in summaries")
    parser.add_argument("--object-limit", type=int, default=5, help="How many objects to include in summaries")
    parser.add_argument("--reuse", action="store_true", help="Reuse generated corpora for supported document types")
    parser.add_argument("--parser", default="builtin", help="Parser to use. Default: builtin")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    app = _load_app(
        documents=[],
        corpora=[],
        folder=args.folder,
        include=args.include,
        exclude=args.exclude,
        reuse=args.reuse,
        parser=args.parser,
    )
    payload = _build_workflow_payload(
        app,
        query=args.query,
        object_query=args.query,
        object_type=None,
        top_k=args.top_k,
        page_limit=args.page_limit,
        object_limit=args.object_limit,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
