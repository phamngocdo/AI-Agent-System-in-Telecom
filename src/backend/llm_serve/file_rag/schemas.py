from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document


@dataclass(frozen=True)
class UploadedDocument:
    file_id: str
    filename: str
    file_type: str
    chunk_count: int


@dataclass
class RetrievedChunk:
    point_id: str
    file_id: str
    filename: str
    chunk_index: int
    text: str
    dense_score: float
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: Optional[float] = None

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "point_id": self.point_id,
                "file_id": self.file_id,
                "filename": self.filename,
                "chunk_index": self.chunk_index,
                "dense_score": self.dense_score,
                "bm25_score": self.bm25_score,
                "combined_score": self.combined_score,
            },
        )
