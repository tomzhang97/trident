"""Simple retrieval utilities used by TRIDENT pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .candidates import Passage

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - scikit-learn optional
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore


@dataclass
class RetrievalResult:
    passages: List[Passage]
    scores: List[float]


class SimpleRetriever:
    """TF-IDF based retriever with graceful fallback."""

    def __init__(self, documents: Sequence[str], ids: Sequence[str] | None = None) -> None:
        self.documents = list(documents)
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(documents))]
        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        else:  # pragma: no cover
            self.vectorizer = None
            self.doc_vectors = None

    def query(self, text: str, top_k: int = 20, cost_per_token: int = 4) -> RetrievalResult:
        if self.vectorizer is None:
            tokens = text.lower().split()
            scores = []
            for doc in self.documents:
                overlap = sum(1 for token in tokens if token in doc.lower())
                scores.append(overlap)
        else:
            query_vec = self.vectorizer.transform([text])
            similarities = cosine_similarity(query_vec, self.doc_vectors)[0]
            scores = similarities.tolist()
        scored = sorted(zip(self.ids, self.documents, scores), key=lambda x: x[2], reverse=True)[:top_k]
        passages = [
            Passage(pid=item[0], text=item[1], cost=len(item[1].split()) * cost_per_token)
            for item in scored
        ]
        return RetrievalResult(passages=passages, scores=[score for *_, score in scored])


def chunk_documents(documents: Iterable[str], chunk_size: int = 150) -> List[str]:
    """Break long documents into fixed-size token windows."""

    chunks: List[str] = []
    for doc in documents:
        words = doc.split()
        for idx in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[idx : idx + chunk_size]))
    return chunks
