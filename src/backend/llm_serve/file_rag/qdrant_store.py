from typing import Sequence
from uuid import uuid4

from qdrant_client import models

from core.config import settings
from qdrant import get_qdrant_client


class QdrantChunkStore:
    def __init__(self) -> None:
        self.collection_name = settings.FILE_RAG_COLLECTION_NAME or settings.QDRANT_COLLECTION_NAME

    async def upsert_chunks(
        self,
        *,
        vectors: Sequence[Sequence[float]],
        chunks: Sequence[str],
        file_id: str,
        session_id: str,
        user_id: str,
        filename: str,
        file_type: str,
    ) -> None:
        if not vectors:
            raise RuntimeError("Embedding model không trả về vector.")
        if len(vectors) != len(chunks):
            raise RuntimeError("Số lượng vector và chunk không khớp.")

        await self.ensure_collection(len(vectors[0]))
        client = get_qdrant_client()
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=list(vector),
                payload={
                    "file_id": file_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        await client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    async def search(
        self,
        *,
        vector: Sequence[float],
        file_ids: Sequence[str],
        session_id: str,
        user_id: str,
        limit: int,
    ):
        client = get_qdrant_client()
        query_filter = self._search_filter(
            file_ids=file_ids,
            session_id=session_id,
            user_id=user_id,
        )

        if hasattr(client, "search"):
            return await client.search(
                collection_name=self.collection_name,
                query_vector=list(vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        response = await client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return getattr(response, "points", response)

    async def list_chunks(
        self,
        *,
        file_ids: Sequence[str],
        session_id: str,
        user_id: str,
    ):
        client = get_qdrant_client()
        query_filter = self._search_filter(
            file_ids=file_ids,
            session_id=session_id,
            user_id=user_id,
        )
        offset = None
        records = []

        while True:
            response = await client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=settings.FILE_RAG_SCROLL_BATCH_SIZE,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            points, offset = self._unpack_scroll_response(response)
            records.extend(points)
            if offset is None:
                return records

    async def delete_session_chunks(self, *, session_id: str, user_id: str) -> None:
        client = get_qdrant_client()
        await client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=self._session_user_filter(session_id=session_id, user_id=user_id)
            ),
            wait=True,
        )

    async def ensure_collection(self, vector_size: int) -> None:
        client = get_qdrant_client()
        try:
            await client.get_collection(self.collection_name)
            return
        except Exception:
            pass

        try:
            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            await self._create_payload_index("file_id")
            await self._create_payload_index("session_id")
            await self._create_payload_index("user_id")
        except Exception:
            await client.get_collection(self.collection_name)

    async def _create_payload_index(self, field_name: str) -> None:
        client = get_qdrant_client()
        try:
            await client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    @staticmethod
    def _search_filter(
        *,
        file_ids: Sequence[str],
        session_id: str,
        user_id: str,
    ) -> models.Filter:
        must_conditions = QdrantChunkStore._session_user_filter(
            session_id=session_id,
            user_id=user_id,
        ).must or []
        file_conditions = [
            models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))
            for file_id in file_ids
        ]
        if len(file_conditions) == 1:
            return models.Filter(must=[*must_conditions, *file_conditions])
        return models.Filter(must=must_conditions, should=file_conditions)

    @staticmethod
    def _session_user_filter(*, session_id: str, user_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                ),
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
            ]
        )

    @staticmethod
    def _unpack_scroll_response(response) -> tuple[list, object]:
        if isinstance(response, tuple):
            return list(response[0]), response[1]

        points = getattr(response, "points", response)
        offset = getattr(response, "next_page_offset", None)
        return list(points), offset
