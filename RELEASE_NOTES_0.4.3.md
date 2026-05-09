## MARE v0.4.3

MARE `v0.4.3` builds on the product cleanup in `v0.4.2` and makes the public package match the repo's newer document and agent workflows more honestly.

This release is about tightening the real product story:

- grounded retrieval across more than just PDFs
- better mixed-document local workflows
- stronger agent-facing proof delivery over MCP
- cleaner public examples and release packaging

### Highlights

- Expanded first-pass local document support beyond PDFs:
  - `pdf`
  - `md` / `markdown`
  - `txt`
  - `docx`
- Added mixed-document local workflows:
  - `mare chat --folder ./examples/mixed_docs`
  - `mare workflow --folder ./examples/mixed_docs --query "..."`
  - `examples/mixed_docs_workflow.py`
- Added public mixed-document sample content in `examples/mixed_docs/` for repeatable demos and tests.
- Improved local folder discovery for `mare chat` and `mare workflow` so mixed document trees work more naturally.
- Expanded object extraction and evidence handling for markdown, text, and docx-backed corpora.
- Improved Streamlit UI stability and public-facing copy:
  - safer missing-image handling
  - clearer document upload story
  - cleaner developer playground notebook examples
- Improved MCP evidence delivery for agents and remote app clients:
  - `ingest_document`
  - `query_document`
  - `best_evidence`
  - `proof_assets`
  - `primary_proof_asset`
  - `proof_links`
  - public page/highlight asset URLs when `mare mcp` is started with a public base URL
- Cleaned the public repo boundary:
  - better separation of local artifacts vs public examples
  - packaging now includes the shipped example scripts and mixed-document samples

### Why this release matters

`v0.4.2` made MARE feel more like one product.

`v0.4.3` makes that product story more truthful:

- MARE is no longer only a PDF-shaped local workflow
- the repo now ships runnable mixed-document examples
- the MCP layer is better suited to agent workflows that need grounded proof, not just snippets

The core model stays the same:

```text
user or agent -> MARE -> citation + snippet + proof
```

What changed is the breadth and polish of the surfaces around that loop.

### Install

```bash
pip install mare-retrieval
```

UI:

```bash
pip install "mare-retrieval[ui]"
mare ui
```

MCP:

```bash
pip install "mare-retrieval[mcp]"
mare mcp
```

Mixed-document example:

```bash
mare chat --folder ./examples/mixed_docs
```

### Notes

- Python import remains `import mare`
- GitHub repo remains `MARE`
- PyPI distribution name remains `mare-retrieval`
- PDFs still provide the strongest visual proof experience
- markdown, text, and docx are now part of the public local workflow story
- `mare mcp` remains the main protocol-facing integration surface for agents and app platforms

### Docs

- GitHub: https://github.com/mare-retrieval/MARE
- PyPI: https://pypi.org/project/mare-retrieval/
