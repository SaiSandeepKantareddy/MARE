# MARE

MARE is a small open-source starting point for modality-aware retrieval.

It is inspired by the direction highlighted in the IRPAPERS paper, which shows that page-image retrieval and text retrieval have complementary failure modes on scientific documents. Instead of flattening everything into one retrieval path, MARE treats routing, retrieval, fusion, and observability as separate system concerns.

## What this repo is

- A lightweight retrieval layer between a query and modality-specific indexes
- A baseline router that decides whether a query should hit text, image, layout, or a hybrid path
- A late-fusion layer that combines modality-specific scores
- An explainable debug surface that tells you why a modality was selected

## What this repo is not

- Not a chatbot wrapper
- Not a full PDF parsing stack yet
- Not a claim that heuristic routing is state of the art

## Why now

IRPAPERS asks a useful systems question: when should we retrieve over OCR text, page images, layout structure, or some combination? The paper reports that text-based and image-based retrieval each solve queries the other misses, and that fusion improves retrieval quality over either modality alone.

This repo turns that observation into an MVP developer layer.

Paper: https://arxiv.org/pdf/2602.17687

## Architecture

```text
query
  -> router
  -> modality-specific retrievers
     -> text index
     -> image index
     -> layout index
  -> fusion
  -> explainable results
```

Current implementation choices:

- Router: keyword heuristic baseline
- Text retrieval: token-overlap cosine baseline
- Image retrieval: caption and visual-tag overlap baseline
- Layout retrieval: layout-hint overlap baseline
- Fusion: weighted late fusion

The point of `v0.1` is not raw benchmark quality. It is to package the control plane cleanly enough that stronger models can drop in later.

## Repo layout

```text
src/mare/
  engine.py
  router.py
  fusion.py
  types.py
  retrievers/
examples/
tests/
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
mare-demo --query "show me the architecture diagram of transformer"
```

Or without installing the package yet:

```bash
PYTHONPATH=src python3 -m mare.demo --query "show me the architecture diagram of transformer"
```

## Ingest a real PDF

You can convert a PDF into a page-level JSON corpus and then run retrieval on it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
mare-ingest "MacBook Pro (14-inch, M5 Pro or M5 Max) MagSafe 3 Board - Apple Support.pdf"
mare-demo --corpus "generated/MacBook Pro (14-inch, M5 Pro or M5 Max) MagSafe 3 Board - Apple Support.json" --query "what does MagSafe 3 refer to"
```

Without installing the package first:

```bash
PYTHONPATH=src python3 -m mare.ingest "MacBook Pro (14-inch, M5 Pro or M5 Max) MagSafe 3 Board - Apple Support.pdf"
PYTHONPATH=src python3 -m mare.demo --corpus "generated/MacBook Pro (14-inch, M5 Pro or M5 Max) MagSafe 3 Board - Apple Support.json" --query "what does MagSafe 3 refer to"
```

What the ingest step does right now:

- reads each PDF page with `pypdf`
- extracts page text
- creates one retrieval document per page
- adds lightweight layout hints when terms like `Table` or `Figure` appear
- writes a JSON corpus that the retriever can search immediately

This is intentionally a text-first ingest path. Image extraction, OCR, page thumbnails, and real layout modeling are the next step.

Example output:

```json
{
  "query": "show me the architecture diagram of transformer",
  "intent": "visual_lookup",
  "selected_modalities": ["image"],
  "discarded_modalities": ["text", "layout"],
  "confidence": 0.8,
  "rationale": "Detected modality cues in query tokens. Selected image based on keyword overlap with routing hints.",
  "results": [
    {
      "doc_id": "paper-transformer-p4",
      "title": "Attention Is All You Need",
      "page": 4,
      "score": 0.6,
      "reason": "image:Matched visual cues: architecture, diagram, transformer"
    }
  ]
}
```

## Why the explainability matters

The debug surface is a core feature, not an afterthought. For production retrieval systems, we need to answer:

- Which modality did the router choose?
- Which modalities were skipped?
- Why did a page rank highly?
- What tradeoff did fusion make?

That is the wedge for MARE: make multimodal retrieval inspectable before trying to make it magical.

## Local sample data

`examples/sample_corpus.json` contains a tiny IR-paper-style corpus so the routing and fusion path is runnable out of the box.

There is also a local PDF in this workspace:

- `MacBook Pro (14-inch, M5 Pro or M5 Max) MagSafe 3 Board - Apple Support.pdf`

That file can now be ingested into a JSON page corpus with `mare-ingest`.

## Roadmap

### v0.1

- text + image + layout routing
- weighted late fusion
- explainable retrieval output
- tests and runnable demo

### v0.2

- pluggable embedding backends
- PDF page ingestion
- OCR and caption extraction adapters
- score normalization per modality

### v0.3

- learned router
- benchmark harness for IRPAPERS-style evaluation
- cost-aware routing budgets
- reranking and cross-modal evidence aggregation

## Suggested next open-source moves

- Add adapters for FAISS, Qdrant, and Weaviate
- Add page extraction from PDFs
- Add a benchmark runner that computes Recall@k per modality
- Add a small web debug UI for route inspection

## License

MIT
