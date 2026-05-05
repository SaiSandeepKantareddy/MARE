from __future__ import annotations

import argparse
import json
from pathlib import Path

from mare.api import MAREApp
from mare.workflow import _build_workflow_payload, _load_app


def _discover_folder_inputs(folder: Path) -> tuple[list[str], list[str]]:
    pdfs = sorted(str(path) for path in folder.glob("*.pdf"))
    corpora = sorted(str(path) for path in folder.glob("*.json"))
    return pdfs, corpora


def _build_app_from_args(
    *,
    folder: str | None,
    pdfs: list[str],
    corpora: list[str],
    reuse: bool,
    parser: str,
) -> MAREApp:
    resolved_pdfs = list(pdfs)
    resolved_corpora = list(corpora)
    if folder:
        folder_pdfs, folder_corpora = _discover_folder_inputs(Path(folder))
        resolved_pdfs.extend(folder_pdfs)
        resolved_corpora.extend(folder_corpora)
    return _load_app(pdfs=resolved_pdfs, corpora=resolved_corpora, reuse=reuse, parser=parser)


def _print_intro(app: MAREApp) -> None:
    source_pdfs = [str(path) for path in app.source_pdfs]
    print("MARE Chat")
    print("")
    if source_pdfs:
        print(f"Loaded PDFs: {', '.join(source_pdfs)}")
    if app.corpus_paths:
        print(f"Loaded corpora: {', '.join(str(path) for path in app.corpus_paths)}")
    print("Type a question, or use :help, :sources, :json <question>, :quit")
    print("")


def _print_help() -> None:
    print("Commands")
    print(":help      Show this help")
    print(":sources   Show loaded PDFs and corpora")
    print(":json ...  Return the workflow payload as JSON for the rest of the line")
    print(":quit      Exit mare-chat")
    print("")


def _print_sources(app: MAREApp) -> None:
    print("Sources")
    if app.source_pdfs:
        print(f"PDFs: {', '.join(str(path) for path in app.source_pdfs)}")
    else:
        print("PDFs: none")
    if app.corpus_paths:
        print(f"Corpora: {', '.join(str(path) for path in app.corpus_paths)}")
    else:
        print("Corpora: none")
    print("")


def _print_answer(payload: dict) -> None:
    results = payload["steps"]["query_corpus"]["results"]
    print(f"Question: {payload['steps']['query_corpus']['query']}")
    if not results:
        print("Answer: No matching evidence found.")
        print("")
        return

    best = results[0]
    source_pdf = best.get("metadata", {}).get("source", "")
    print(f"Best page: {best['page']}")
    if source_pdf:
        print(f"Source PDF: {source_pdf}")
    print(f"Object type: {best['object_type'] or 'page'}")
    print(f"Snippet: {best['snippet'] or '[no snippet available]'}")
    print(f"Reason: {best['reason']}")
    print(f"Page image: {best['page_image_path'] or '[no page image available]'}")
    print(f"Highlight: {best['highlight_image_path'] or '[no highlight available]'}")
    print("")


def run_chat(app: MAREApp, *, top_k: int = 3, page_limit: int = 3, object_limit: int = 5) -> None:
    _print_intro(app)
    while True:
        try:
            raw = input("mare> ").strip()
        except EOFError:
            print("")
            break
        except KeyboardInterrupt:
            print("")
            break

        if not raw:
            continue
        if raw in {":quit", ":exit"}:
            break
        if raw == ":help":
            _print_help()
            continue
        if raw == ":sources":
            _print_sources(app)
            continue
        if raw.startswith(":json"):
            query = raw[len(":json") :].strip()
            if not query:
                print("Usage: :json <question>")
                print("")
                continue
            payload = _build_workflow_payload(
                app,
                query=query,
                object_query=query,
                object_type=None,
                top_k=top_k,
                page_limit=page_limit,
                object_limit=object_limit,
            )
            print(json.dumps(payload, indent=2))
            print("")
            continue

        payload = _build_workflow_payload(
            app,
            query=raw,
            object_query=raw,
            object_type=None,
            top_k=top_k,
            page_limit=page_limit,
            object_limit=object_limit,
        )
        _print_answer(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple document evidence chat loop over PDFs or corpora")
    parser.add_argument("--folder", help="Folder containing PDFs and/or MARE corpus JSON files")
    parser.add_argument("--pdf", action="append", default=[], help="Path to a PDF. Repeat to load multiple PDFs.")
    parser.add_argument(
        "--corpus",
        dest="corpora",
        action="append",
        default=[],
        help="Path to an existing MARE corpus JSON. Repeat to load multiple corpora.",
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse generated corpora for PDFs when available")
    parser.add_argument("--parser", default="builtin", help="Parser to use for --pdf ingestion. Default: builtin")
    parser.add_argument("--top-k", type=int, default=3, help="How many retrieval hits to consider")
    parser.add_argument("--page-limit", type=int, default=3, help="How many pages to include in summaries")
    parser.add_argument("--object-limit", type=int, default=5, help="How many objects to include in summaries")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    app = _build_app_from_args(
        folder=args.folder,
        pdfs=args.pdf,
        corpora=args.corpora,
        reuse=args.reuse,
        parser=args.parser,
    )
    run_chat(app, top_k=args.top_k, page_limit=args.page_limit, object_limit=args.object_limit)


if __name__ == "__main__":
    main()
