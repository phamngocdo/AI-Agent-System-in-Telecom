from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    PROJECT_NAME: str = "TelcoLLM Backend"
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "TelcoLLM_ai"
    VLLM_URL: str = "http://localhost:8001/v1"
    VLLM_API_KEY: Optional[str] = None
    VLLM_MODEL: str = "TelcoLLM"
    VLLM_TIMEOUT_SECONDS: float = 60.0
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "telco_documents"
    QDRANT_TIMEOUT_SECONDS: float = 10.0
    FILE_RAG_COLLECTION_NAME: Optional[str] = None
    FILE_RAG_CHUNK_SIZE: int = 1200
    FILE_RAG_CHUNK_OVERLAP: int = 180
    FILE_RAG_SEARCH_LIMIT_PER_QUERY: int = 12
    FILE_RAG_RERANK_CANDIDATES: int = 16
    FILE_RAG_CONTEXT_CHUNKS: int = 6
    FILE_RAG_CONTEXT_MAX_CHARS: int = 12000
    FILE_RAG_DENSE_WEIGHT: float = 0.70
    FILE_RAG_BM25_WEIGHT: float = 0.30
    FILE_RAG_BM25_K1: float = 1.5
    FILE_RAG_BM25_B: float = 0.75
    FILE_RAG_SCROLL_BATCH_SIZE: int = 256
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_BATCH_SIZE: int = 16
    RERANK_MODEL: Optional[str] = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANK_DEVICE: str = "cpu"
    SECRET_KEY: str = "a_very_secret_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, env_file_encoding="utf-8", extra="ignore")


settings = Settings()
