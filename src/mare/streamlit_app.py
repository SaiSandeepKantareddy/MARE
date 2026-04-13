from __future__ import annotations

import tempfile
from pathlib import Path


def _require_streamlit():
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "streamlit is required for the visual demo. Install it with `pip install -e '.[ui]'` "
            "or `pip install streamlit`."
        ) from exc
    return st


def main() -> None:
    st = _require_streamlit()

    from mare.ask import ask_pdf

    st.set_page_config(page_title="MARE Demo", layout="wide")
    st.title("MARE")
    st.caption("Ask a PDF a question and see the best page, snippet, and evidence image.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    with st.form("mare_query_form"):
        query = st.text_input("Ask a question about the document")
        top_k = st.slider("How many results to show", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Ask MARE")

    if uploaded_pdf is None:
        st.info("Upload a PDF to start.")
        return

    temp_dir = Path(tempfile.gettempdir()) / "mare_streamlit"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getvalue())

    if submitted:
        if not query.strip():
            st.warning("Enter a question first.")
            return

        with st.spinner("Ingesting PDF and retrieving best pages..."):
            corpus_path, explanation = ask_pdf(pdf_path=pdf_path, query=query, top_k=top_k, reuse=False)

        if not explanation.fused_results:
            st.error("No matching page found.")
            return

        best = explanation.fused_results[0]
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Best Match")
            st.markdown(f"**Page:** {best.page}")
            st.markdown(f"**Score:** {best.score}")
            st.markdown(f"**Reason:** {best.reason}")
            st.markdown("**Snippet:**")
            st.write(best.snippet or "[no snippet available]")

        with right:
            st.subheader("Evidence Image")
            image_path = Path(best.highlight_image_path or best.page_image_path)
            if image_path.exists():
                caption = f"Highlighted page {best.page}" if best.highlight_image_path else f"Page {best.page}"
                st.image(str(image_path), caption=caption)
            else:
                st.warning("No page image available.")

        if len(explanation.fused_results) > 1:
            st.subheader("Other Candidate Pages")
            for hit in explanation.fused_results[1:]:
                st.write(
                    {
                        "page": hit.page,
                        "score": hit.score,
                        "reason": hit.reason,
                        "snippet": hit.snippet,
                        "page_image_path": hit.page_image_path,
                    }
                )

        with st.expander("Debug details"):
            st.write(
                {
                    "intent": explanation.plan.intent,
                    "selected_modalities": [item.value for item in explanation.plan.selected_modalities],
                    "discarded_modalities": [item.value for item in explanation.plan.discarded_modalities],
                    "confidence": explanation.plan.confidence,
                    "rationale": explanation.plan.rationale,
                    "corpus": str(corpus_path),
                }
            )


if __name__ == "__main__":
    main()
