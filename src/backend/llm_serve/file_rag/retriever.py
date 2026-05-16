import logging
from typing import Optional, Sequence

from core.config import settings
from .bm25 import BM25Index
from .constants import REWRITE_QUERY_PROMPT
from .hf_models import LocalCrossEncoderReranker, LocalEmbeddingModel
from .qdrant_store import QdrantChunkStore
from .schemas import RetrievedChunk
from .utils import dedupe, format_history, parse_json_string_list, tokenize_terms


logger = logging.getLogger("telcollm.file_rag")


class DocumentRetriever:
    def __init__(
        self,
        *,
        embeddings: LocalEmbeddingModel,
        chunk_store: QdrantChunkStore,
        reranker: LocalCrossEncoderReranker,
    ) -> None:
        self.embeddings = embeddings
        self.chunk_store = chunk_store
        self.reranker = reranker

    async def build_context(
        self,
        *,
        session_id: str,
        user_id: str,
        telco_llm,
        question: str,
        file_ids: Sequence[str],
        conversation_history: Optional[Sequence[tuple[str, str]]] = None,
    ) -> Optional[str]:
        normalized_file_ids = dedupe([file_id for file_id in file_ids if file_id])
        if not normalized_file_ids:
            logger.info(
                "RAG skipped: no file_ids | session_id=%s user_id=%s question=%r",
                session_id,
                user_id,
                question,
            )
            return None

        logger.info(
            "RAG build start | session_id=%s user_id=%s files=%d question=%r",
            session_id,
            user_id,
            len(normalized_file_ids),
            question,
        )
        source_records = await self.chunk_store.list_chunks(
            file_ids=normalized_file_ids,
            session_id=session_id,
            user_id=user_id,
        )
        if not source_records:
            logger.info(
                "RAG skipped: no stored chunks | session_id=%s user_id=%s files=%d",
                session_id,
                user_id,
                len(normalized_file_ids),
            )
            return None

        if len(source_records) <= settings.FILE_RAG_CONTEXT_CHUNKS:
            chunks = self._records_to_chunks(source_records, normalized_file_ids)
            context, included_chunks, context_chars = self._format_context_with_stats(chunks)
            logger.info(
                "RAG direct context | session_id=%s chunks=%d context_chunks=%d context_chars=%d",
                session_id,
                len(source_records),
                included_chunks,
                context_chars,
            )
            return context

        queries = await self._rewrite_queries(telco_llm, question, conversation_history)
        logger.info(
            "RAG rewritten queries | session_id=%s queries=%s",
            session_id,
            queries,
        )
        candidates = await self._hybrid_search(
            queries=queries,
            file_ids=normalized_file_ids,
            session_id=session_id,
            user_id=user_id,
            source_records=source_records,
        )
        if not candidates:
            logger.info(
                "RAG no candidates | session_id=%s user_id=%s files=%d queries=%d",
                session_id,
                user_id,
                len(normalized_file_ids),
                len(queries),
            )
            return None

        reranked = await self._rerank_if_needed(
            question=question,
            candidates=candidates,
            session_id=session_id,
        )
        context, included_chunks, context_chars = self._format_context_with_stats(
            reranked[: settings.FILE_RAG_CONTEXT_CHUNKS]
        )
        logger.info(
            "RAG context ready | session_id=%s candidates=%d reranked=%d context_chunks=%d context_chars=%d top_chunks=%s",
            session_id,
            len(candidates),
            len(reranked),
            included_chunks,
            context_chars,
            self._chunk_summaries(reranked[:included_chunks]),
        )
        return context

    async def _rewrite_queries(
        self,
        telco_llm,
        question: str,
        conversation_history: Optional[Sequence[tuple[str, str]]],
    ) -> list[str]:
        history_text = format_history(conversation_history)
        user_prompt = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Question:\n{question}\n\n"
            "JSON array:"
        )
        try:
            response = await telco_llm.generate_text(
                system_prompt=REWRITE_QUERY_PROMPT,
                user_prompt=user_prompt,
                temperature=0.7,
                think=False,
            )
            queries = parse_json_string_list(response)
        except Exception:
            logger.exception("RAG query rewrite failed; falling back to template queries.")
            queries = []

        queries = dedupe([question, *queries])
        while len(queries) < 3:
            if len(queries) == 1:
                queries.append(f"Nội dung liên quan trực tiếp đến: {question}")
            else:
                queries.append(f"Thông tin kỹ thuật, định nghĩa và số liệu cho câu hỏi: {question}")
        return queries[:3]

    async def _hybrid_search(
        self,
        *,
        queries: Sequence[str],
        file_ids: Sequence[str],
        session_id: str,
        user_id: str,
        source_records: Sequence[object],
    ) -> list[RetrievedChunk]:
        query_vectors = await self.embeddings.embed_documents(list(queries))
        candidates: dict[str, RetrievedChunk] = {}

        dense_match_count = 0
        for query, vector in zip(queries, query_vectors):
            points = await self.chunk_store.search(
                vector=vector,
                file_ids=file_ids,
                session_id=session_id,
                user_id=user_id,
                limit=settings.FILE_RAG_SEARCH_LIMIT_PER_QUERY,
            )
            point_count = len(points)
            dense_match_count += point_count
            logger.info(
                "RAG dense search | session_id=%s query=%r matches=%d",
                session_id,
                query,
                point_count,
            )
            for point in points:
                self._merge_candidate(candidates, self._point_to_chunk(point))

        bm25_chunks = self._bm25_search(
            queries=queries,
            records=source_records,
            session_id=session_id,
        )
        for chunk in bm25_chunks:
            self._merge_candidate(candidates, chunk)

        ranked = list(candidates.values())
        self._apply_hybrid_scores(ranked)
        ranked = sorted(ranked, key=lambda chunk: chunk.combined_score, reverse=True)[
            : settings.FILE_RAG_RERANK_CANDIDATES
        ]
        logger.info(
            "RAG hybrid search | session_id=%s dense_matches=%d bm25_matches=%d merged_candidates=%d returned_candidates=%d",
            session_id,
            dense_match_count,
            len(bm25_chunks),
            len(candidates),
            len(ranked),
        )
        return ranked

    def _bm25_search(
        self,
        *,
        queries: Sequence[str],
        records: Sequence[object],
        session_id: str,
    ) -> list[RetrievedChunk]:
        if not records:
            logger.info("RAG BM25 skipped: no stored chunks | session_id=%s", session_id)
            return []

        logger.info(
            "RAG BM25 index | session_id=%s records=%d queries=%d",
            session_id,
            len(records),
            len(queries),
        )
        index = BM25Index(
            records,
            [tokenize_terms(str((record.payload or {}).get("text") or "")) for record in records],
        )
        chunks: dict[str, RetrievedChunk] = {}
        for query in queries:
            query_tokens = tokenize_terms(query)
            matches = index.search(query_tokens, settings.FILE_RAG_SEARCH_LIMIT_PER_QUERY)
            logger.info(
                "RAG BM25 search | session_id=%s query=%r tokens=%d matches=%d",
                session_id,
                query,
                len(query_tokens),
                len(matches),
            )
            for match in matches:
                chunk = self._point_to_chunk(match.document, bm25_score=match.score)
                existing = chunks.get(chunk.point_id)
                if existing is None or existing.bm25_score < chunk.bm25_score:
                    chunks[chunk.point_id] = chunk
        return list(chunks.values())

    async def _rerank_if_needed(
        self,
        *,
        question: str,
        candidates: list[RetrievedChunk],
        session_id: str,
    ) -> list[RetrievedChunk]:
        if len(candidates) <= settings.FILE_RAG_CONTEXT_CHUNKS:
            logger.info(
                "RAG rerank skipped | session_id=%s candidates=%d context_chunks=%d",
                session_id,
                len(candidates),
                settings.FILE_RAG_CONTEXT_CHUNKS,
            )
            return candidates

        return await self.reranker.rerank(question, candidates)

    @staticmethod
    def _point_to_chunk(point, *, bm25_score: float = 0.0) -> RetrievedChunk:
        payload = point.payload or {}
        text = str(payload.get("text") or "")
        return RetrievedChunk(
            point_id=str(point.id),
            file_id=str(payload.get("file_id") or ""),
            filename=str(payload.get("filename") or "uploaded.pdf"),
            chunk_index=int(payload.get("chunk_index") or 0),
            text=text,
            dense_score=float(getattr(point, "score", 0.0) or 0.0),
            bm25_score=bm25_score,
        )

    @classmethod
    def _records_to_chunks(
        cls,
        records: Sequence[object],
        file_ids: Sequence[str],
    ) -> list[RetrievedChunk]:
        file_order = {file_id: index for index, file_id in enumerate(file_ids)}
        chunks = [cls._point_to_chunk(record) for record in records]
        return sorted(
            chunks,
            key=lambda chunk: (
                file_order.get(chunk.file_id, len(file_order)),
                chunk.chunk_index,
            ),
        )

    @staticmethod
    def _merge_candidate(candidates: dict[str, RetrievedChunk], chunk: RetrievedChunk) -> None:
        existing = candidates.get(chunk.point_id)
        if existing is None:
            candidates[chunk.point_id] = chunk
            return

        existing.dense_score = max(existing.dense_score, chunk.dense_score)
        existing.bm25_score = max(existing.bm25_score, chunk.bm25_score)

    @staticmethod
    def _apply_hybrid_scores(chunks: list[RetrievedChunk]) -> None:
        if not chunks:
            return

        max_dense = max((chunk.dense_score for chunk in chunks), default=0.0) or 1.0
        max_bm25 = max((chunk.bm25_score for chunk in chunks), default=0.0) or 1.0
        for chunk in chunks:
            dense = chunk.dense_score / max_dense
            bm25 = chunk.bm25_score / max_bm25
            chunk.combined_score = (
                settings.FILE_RAG_DENSE_WEIGHT * dense
                + settings.FILE_RAG_BM25_WEIGHT * bm25
            )

    @staticmethod
    def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
        context, _, _ = DocumentRetriever._format_context_with_stats(chunks)
        return context

    @staticmethod
    def _format_context_with_stats(chunks: Sequence[RetrievedChunk]) -> tuple[str, int, int]:
        blocks = []
        total_chars = 0
        for index, chunk in enumerate(chunks, start=1):
            block = (
                f"[File excerpt {index} | file_id={chunk.file_id} | "
                f"filename={chunk.filename} | chunk={chunk.chunk_index}]\n"
                f"{chunk.text.strip()}"
            )
            if total_chars + len(block) > settings.FILE_RAG_CONTEXT_MAX_CHARS:
                break
            blocks.append(block)
            total_chars += len(block)

        return "\n\n---\n\n".join(blocks), len(blocks), total_chars

    @staticmethod
    def _chunk_summaries(chunks: Sequence[RetrievedChunk]) -> list[dict]:
        return [
            {
                "file_id": chunk.file_id,
                "filename": chunk.filename,
                "chunk": chunk.chunk_index,
                "dense": round(chunk.dense_score, 4),
                "bm25": round(chunk.bm25_score, 4),
                "combined": round(chunk.combined_score, 4),
                "rerank": round(chunk.rerank_score, 4) if chunk.rerank_score is not None else None,
            }
            for chunk in chunks
        ]
