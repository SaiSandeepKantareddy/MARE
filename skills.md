# MARE Repo Skills

Internal session bootstrap for this repository.

This file is optimized for recovering context quickly after chat history is gone.
For a cleaner public-facing document, see `DEVELOPER_GUIDE.md`.

This file is a fast spin-up guide for future sessions.

## What This Repo Is

MARE is an evidence-first PDF retrieval library for developers and agents.

Core promise:

- ingest a PDF locally
- retrieve the best page for a question
- return the exact snippet when possible
- return a page image and highlighted evidence image
- expose structured results for apps, agents, and framework adapters

Primary use case today:

- manuals
- support PDFs
- technical documentation

The built-in lexical and object-aware stack is the default and currently the strongest overall path.

## Mental Model

Think of MARE as this pipeline:

`PDF -> corpus JSON + page images -> page objects -> query routing -> retrieval -> fusion/rerank -> evidence payload`

The important design principle is:

`question about a PDF -> grounded evidence -> another system uses that evidence`

The project is not trying to be:

- a general PDF chatbot
- a full agent framework
- a generic vector database
- a full document graph platform

## Most Important Files

### Core API and engine

- `src/mare/api.py`
  - `MAREApp` is the main app wrapper.
  - Best entrypoint for programmatic usage.
- `src/mare/engine.py`
  - `MAREngine` routes query -> modality retrievers -> fusion -> optional reranker.
- `src/mare/types.py`
  - Core dataclasses: `Document`, `DocumentObject`, `RetrievalHit`, `QueryPlan`, `RetrievalExplanation`.

### Ingestion and evidence extraction

- `src/mare/ingest.py`
  - Built-in PDF ingestion using `pypdf` and `pypdfium2`.
  - Produces corpus JSON plus rendered page images.
- `src/mare/objects.py`
  - Extracts `procedure`, `figure`, `table`, and `section` objects from page text.
  - Important for evidence quality, especially manuals.
- `src/mare/highlight.py`
  - Produces text-region highlight images and fallback region overlays for objects.

### Retrieval

- `src/mare/router.py`
  - Heuristic modality router for `text`, `image`, `layout`.
- `src/mare/retrievers/text.py`
  - Main retrieval logic.
  - This is where lexical scoring, object-aware boosts, snippet choice, and highlight behavior come together.
- `src/mare/retrievers/image.py`
  - Lightweight image/layout retrieval.
- `src/mare/fusion.py`
  - Weighted late fusion across modalities.

### Extensions and integrations

- `src/mare/extensions.py`
  - Pluggable parsers, retrievers, rerankers, FAISS/Qdrant helpers.
  - Main extension surface for advanced stacks.
- `src/mare/integrations.py`
  - LangChain, LangGraph, and LlamaIndex adapters.
- `src/mare/mcp_server.py`
  - MCP tool surface for ingest/query/page-objects workflows.

### User-facing tools

- `src/mare/ask.py`
  - Simple CLI for asking a PDF a question.
- `src/mare/demo.py`
  - Demo over an existing corpus.
- `src/mare/eval.py`
  - Evaluation harness and stack comparison.
- `src/mare/streamlit_app.py`
  - Visual playground and advanced stack controls.

## Recommended Starting Points

If you need to understand or change the main product behavior, read in this order:

1. `README.md`
2. `src/mare/api.py`
3. `src/mare/engine.py`
4. `src/mare/retrievers/text.py`
5. `src/mare/objects.py`
6. `src/mare/highlight.py`
7. `tests/test_engine.py`
8. `tests/test_objects.py`

If you need extension behavior, then read:

1. `src/mare/extensions.py`
2. `tests/test_extensibility.py`
3. `examples/advanced_stack.py`

If you need the agent/tooling surface, then read:

1. `src/mare/mcp_server.py`
2. `src/mare/integrations.py`
3. `tests/test_mcp_server.py`
4. `tests/test_integrations.py`

## Public Usage Shapes

### Python API

Typical usage:

```python
from mare import MAREApp

app = MAREApp.from_pdf("manual.pdf", reuse=True)
best = app.best_match("how do I connect the AC adapter")
```

### CLI

Ask a PDF directly:

```bash
python3 ask.py "manual.pdf" "how do I connect the AC adapter"
```

Ingest a PDF:

```bash
mare-ingest "manual.pdf"
```

Run demo over a corpus:

```bash
mare-demo --corpus generated/manual.json --query "show me the comparison table"
```

Run eval:

```bash
PYTHONPATH=src python3 -m mare.eval --corpus generated/manual.json --eval examples/manual_116441_eval_cases.json
```

Run UI:

```bash
mare-ui
```

Run MCP server:

```bash
mare-mcp
```

## What Matters Most In Retrieval Quality

The built-in stack is intentionally not just plain keyword search.

Current strengths:

- object-aware scoring
- procedure-aware retrieval for manuals
- grouped procedure extraction when headings exist
- phrase-aware lexical matching
- table and figure boosts
- returning snippets and highlight images as part of the result contract

When debugging retrieval quality, inspect first:

- extracted page text
- extracted objects on the winning page
- `document.metadata["signals"]`
- `document.layout_hints`
- `TextRetriever.retrieve`
- whether the chosen snippet came from a page or a matched object

## Extension Points

The central extension hook is `MAREConfig` in `src/mare/extensions.py`.

Main knobs:

- `parser`
- `retriever_factories`
- `reranker`

Important built-in extension classes:

- parsers:
  - `BuiltinPDFParser`
  - `DoclingParser`
  - `UnstructuredParser`
  - `PaddleOCRParser`
  - `SuryaParser`
- retrievers:
  - `SentenceTransformersRetriever`
  - `HybridSemanticRetriever`
  - `FAISSRetriever`
  - `QdrantHybridRetriever`
- rerankers:
  - `IdentityReranker`
  - `KeywordBoostReranker`
  - `FastEmbedReranker`
- indexers:
  - `FAISSIndexer`
  - `QdrantIndexer`

Guideline:

- prefer the built-in path unless the task clearly needs OCR-heavy parsing, semantic lift, or an external vector backend

## Important Tests

Tests are a good map of intended behavior.

Highest-signal files:

- `tests/test_engine.py`
  - routing and evidence ranking expectations
- `tests/test_objects.py`
  - object extraction behavior
- `tests/test_ingest.py`
  - corpus-building expectations
- `tests/test_extensibility.py`
  - supported integrations and extension hooks
- `tests/test_mcp_server.py`
  - tool payload shape
- `tests/test_integrations.py`
  - framework adapter contracts

Known status from last local pass:

- `pytest -q`
- result: `59 passed`

## Packaging and Release Notes

- package name on PyPI: `mare-retrieval`
- import name: `mare`
- version currently in repo: `0.3.0`

Packaging files:

- `pyproject.toml`
- `setup.py`
- `.github/workflows/publish.yml`
- `PUBLISHING.md`
- `RELEASE_NOTES_0.3.0.md`

If bumping a release, update both:

- `pyproject.toml`
- `setup.py`

## Useful Repo Artifacts

Local example/demo assets in repo:

- `116441.pdf`
- `MacBook Pro ... Apple Support.pdf`
- `generated/`
- `examples/*.json`

These are useful for manual smoke tests and retrieval debugging.

## Practical Session Checklist

When starting fresh:

1. Read `README.md`.
2. Read this file.
3. Run `pytest -q`.
4. Inspect `src/mare/api.py`, `src/mare/engine.py`, and `src/mare/retrievers/text.py`.
5. If the task is retrieval-quality related, inspect `src/mare/objects.py` and `src/mare/highlight.py`.
6. If the task is integrations-related, inspect `src/mare/extensions.py`, `src/mare/integrations.py`, and `src/mare/mcp_server.py`.

## Short Repo Summary

If you only remember one thing:

MARE is a PDF evidence layer. The center of gravity is not answer generation. It is returning inspectable evidence: page, snippet, highlight, object, and rationale.
