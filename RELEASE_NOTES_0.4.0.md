## MARE v0.4.0

MARE is an evidence-first PDF retrieval library for developers and agents.

This release is the point where MARE starts to feel like a real PDF evidence layer rather than only a single-document retrieval demo. The core built-in path is stronger, the agent story is much clearer, and the library can now work across multiple PDFs while preserving grounded page/snippet/highlight output.

### Highlights

- Sharpened product positioning around:
  - evidence-first PDF retrieval
  - page, snippet, highlight, and visual proof
  - developer and agent workflows
- Added a first MCP server layer for agents
  - `mare-mcp`
  - `ingest_pdf`
  - `query_pdf`
  - `query_corpus`
  - `query_corpora`
  - `page_objects`
  - `describe_corpus`
  - `search_objects`
- Added agent-oriented workflow examples
  - `examples/agent_workflow.py`
  - `examples/multi_pdf_workflow.py`
  - `examples/mcp_stdio_config.json`
- Added hybrid semantic retrieval that preserves MARE's lexical/object-aware evidence behavior
- Added object-level semantic matching inside the hybrid retriever
- Improved semantic retrieval so it preserves MARE evidence outputs such as:
  - snippet
  - object type
  - highlight
- Added corpus introspection and object browsing
  - `app.describe_corpus()`
  - `app.search_objects(...)`
- Added multi-PDF retrieval support
  - `MAREApp.from_corpora(...)`
  - `load_corpora(...)`
  - MCP `query_corpora`
- Upgraded the Streamlit playground
  - recommended default stack messaging
  - advanced stack visibility
  - stale-result protection
  - multi-PDF upload and retrieval
- Improved object highlight precision
  - line-aware region metadata for built-in extraction
  - bbox-aware highlighting when richer parsers provide coordinates
  - better region fallback behavior for figures, tables, and sections
- Added stack comparison mode to the eval harness
- Added real benchmark examples for:
  - manuals
  - Apple support docs
  - research papers

### Why this release matters

MARE is now much closer to the intended architecture:

```text
user question -> agent/app -> MARE -> page + snippet + highlight + proof
```

The built-in stack remains the recommended default, while advanced integrations now fit more cleanly as optional infrastructure around the evidence-first core.

### Install

```bash
pip install mare-retrieval
```

Useful extras:

```bash
pip install "mare-retrieval[ui]"
pip install "mare-retrieval[mcp]"
pip install "mare-retrieval[sentence-transformers]"
pip install "mare-retrieval[faiss]"
pip install "mare-retrieval[integrations]"
```

### Notes

- Python import remains `import mare`
- GitHub repo remains `MARE`
- PyPI distribution name remains `mare-retrieval`
- the built-in retrieval path is still the recommended default for most users

### Docs

- GitHub: https://github.com/mare-retrieval/MARE
- PyPI: https://pypi.org/project/mare-retrieval/
