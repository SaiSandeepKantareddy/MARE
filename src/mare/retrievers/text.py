from __future__ import annotations

import math
import re
from collections import Counter

from mare.retrievers.base import BaseRetriever
from mare.types import Modality, RetrievalHit


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _best_snippet(text: str, query: str, window: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    query_terms = [term for term in _tokenize(query) if len(term) > 1]
    if not query_terms:
        return normalized[:window]

    lowered = normalized.lower()
    best_index = -1
    for term in query_terms:
        found = lowered.find(term)
        if found != -1:
            best_index = found if best_index == -1 else min(best_index, found)

    if best_index == -1:
        return normalized[:window]

    start = max(0, best_index - 60)
    end = min(len(normalized), start + window)
    snippet = normalized[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(normalized):
        snippet = snippet + "..."
    return snippet


class TextRetriever(BaseRetriever):
    modality = Modality.TEXT

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        query_counts = Counter(_tokenize(query))
        hits: list[RetrievalHit] = []

        for document in self.documents:
            doc_counts = Counter(_tokenize(document.text))
            overlap = set(query_counts) & set(doc_counts)
            if not overlap:
                continue

            numerator = sum(query_counts[token] * doc_counts[token] for token in overlap)
            norm = math.sqrt(sum(v * v for v in query_counts.values())) * math.sqrt(
                sum(v * v for v in doc_counts.values())
            )
            score = numerator / norm if norm else 0.0
            hits.append(
                RetrievalHit(
                    doc_id=document.doc_id,
                    title=document.title,
                    page=document.page,
                    modality=self.modality,
                    score=score,
                    reason=f"Matched text terms: {', '.join(sorted(overlap)[:5])}",
                    snippet=_best_snippet(document.text, query),
                    page_image_path=document.page_image_path,
                    metadata=document.metadata,
                )
            )

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
