from typing import Optional
from pathlib import Path

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
    SECRET_KEY: str = "a_very_secret_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

settings = Settings()
