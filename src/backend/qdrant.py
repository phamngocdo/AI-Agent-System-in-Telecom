from typing import Optional

from qdrant_client import AsyncQdrantClient

from core.config import settings


class QdrantConnection:
    client: Optional[AsyncQdrantClient] = None

    def connect(self) -> AsyncQdrantClient:
        if self.client is None:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=settings.QDRANT_TIMEOUT_SECONDS,
            )
        return self.client

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None


qdrant = QdrantConnection()


def get_qdrant_client() -> AsyncQdrantClient:
    if qdrant.client is None:
        raise RuntimeError("Qdrant client has not been initialized")
    return qdrant.client
