from __future__ import annotations

from abc import ABC, abstractmethod

from mare.types import Document, Modality, RetrievalHit


class BaseRetriever(ABC):
    modality: Modality

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        raise NotImplementedError

