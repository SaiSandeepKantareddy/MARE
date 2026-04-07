from __future__ import annotations

import math
import re
from collections import Counter

from mare.retrievers.base import BaseRetriever
from mare.types import Modality, RetrievalHit


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


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
                    metadata=document.metadata,
                )
            )

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

