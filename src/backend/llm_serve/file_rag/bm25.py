import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from core.config import settings


@dataclass(frozen=True)
class BM25Match:
    document: object
    score: float


class BM25Index:
    def __init__(self, documents: Sequence[object], tokenized_texts: Sequence[list[str]]) -> None:
        self.documents = list(documents)
        self.term_frequencies = [Counter(tokens) for tokens in tokenized_texts]
        self.doc_lengths = [len(tokens) for tokens in tokenized_texts]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_frequency = self._build_doc_frequency()

    def search(self, query_tokens: Sequence[str], limit: int) -> list[BM25Match]:
        if not query_tokens or not self.documents:
            return []

        scored = [
            BM25Match(document=document, score=score)
            for document, score in zip(self.documents, self._score_documents(query_tokens))
            if score > 0
        ]
        return sorted(scored, key=lambda match: match.score, reverse=True)[:limit]

    def _score_documents(self, query_tokens: Sequence[str]) -> list[float]:
        query_frequency = Counter(query_tokens)
        return [
            self._score_document(term_frequency, doc_length, query_frequency)
            for term_frequency, doc_length in zip(self.term_frequencies, self.doc_lengths)
        ]

    def _score_document(
        self,
        term_frequency: Counter[str],
        doc_length: int,
        query_frequency: Counter[str],
    ) -> float:
        score = 0.0
        for term, query_count in query_frequency.items():
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue

            idf = self._idf(term)
            denominator = frequency + settings.FILE_RAG_BM25_K1 * (
                1 - settings.FILE_RAG_BM25_B
                + settings.FILE_RAG_BM25_B * doc_length / (self.avg_doc_length or 1.0)
            )
            score += query_count * idf * (
                frequency * (settings.FILE_RAG_BM25_K1 + 1) / denominator
            )
        return score

    def _idf(self, term: str) -> float:
        doc_count = len(self.documents)
        frequency = self.doc_frequency.get(term, 0)
        return math.log(1 + (doc_count - frequency + 0.5) / (frequency + 0.5))

    def _build_doc_frequency(self) -> Counter[str]:
        doc_frequency: Counter[str] = Counter()
        for term_frequency in self.term_frequencies:
            doc_frequency.update(term_frequency.keys())
        return doc_frequency
