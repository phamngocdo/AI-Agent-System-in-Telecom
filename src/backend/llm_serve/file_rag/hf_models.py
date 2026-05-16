import asyncio

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import settings
from .schemas import RetrievedChunk


class LocalEmbeddingModel:
    def __init__(self) -> None:
        self.model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": settings.EMBEDDING_NORMALIZE,
                "batch_size": settings.EMBEDDING_BATCH_SIZE,
            },
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.model.embed_documents, texts)


class LocalCrossEncoderReranker:
    def __init__(self) -> None:
        self.reranker = None
        if settings.RERANK_MODEL:
            cross_encoder = HuggingFaceCrossEncoder(
                model_name=settings.RERANK_MODEL,
                model_kwargs={"device": settings.RERANK_DEVICE},
            )
            self.reranker = CrossEncoderReranker(
                model=cross_encoder,
                top_n=settings.FILE_RAG_CONTEXT_CHUNKS,
            )

    async def rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not self.reranker or not chunks:
            return chunks

        documents = [chunk.to_document() for chunk in chunks]
        reranked_documents = await asyncio.to_thread(
            self.reranker.compress_documents,
            documents,
            question,
        )
        order = {
            document.metadata["point_id"]: index
            for index, document in enumerate(reranked_documents)
        }

        ranked_chunks = [
            chunk for chunk in chunks if chunk.point_id in order
        ]
        for chunk in ranked_chunks:
            chunk.rerank_score = float(len(ranked_chunks) - order[chunk.point_id])

        return sorted(
            ranked_chunks,
            key=lambda chunk: (chunk.rerank_score or 0.0, chunk.combined_score),
            reverse=True,
        )
