from __future__ import annotations

import re

from mare.retrievers.base import BaseRetriever
from mare.types import Modality, RetrievalHit


def _shared_terms(query: str, field: str) -> set[str]:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    field_terms = set(re.findall(r"[a-z0-9]+", field.lower()))
    return query_terms & field_terms


class ImageRetriever(BaseRetriever):
    modality = Modality.IMAGE

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []

        for document in self.documents:
            overlap = _shared_terms(query, document.image_caption)
            if not overlap:
                continue

            score = min(1.0, 0.2 * len(overlap))
            hits.append(
                RetrievalHit(
                    doc_id=document.doc_id,
                    title=document.title,
                    page=document.page,
                    modality=self.modality,
                    score=score,
                    reason=f"Matched visual cues: {', '.join(sorted(overlap)[:5])}",
                    snippet=document.image_caption,
                    page_image_path=document.page_image_path,
                    highlight_image_path=document.page_image_path,
                    metadata=document.metadata,
                )
            )

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


class LayoutRetriever(BaseRetriever):
    modality = Modality.LAYOUT

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []

        for document in self.documents:
            overlap = _shared_terms(query, document.layout_hints)
            if not overlap:
                continue

            score = min(1.0, 0.25 * len(overlap))
            hits.append(
                RetrievalHit(
                    doc_id=document.doc_id,
                    title=document.title,
                    page=document.page,
                    modality=self.modality,
                    score=score,
                    reason=f"Matched layout cues: {', '.join(sorted(overlap)[:5])}",
                    snippet=document.layout_hints,
                    page_image_path=document.page_image_path,
                    highlight_image_path=document.page_image_path,
                    metadata=document.metadata,
                )
            )

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
