from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str
    gemini_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "research_documents"

    max_research_iterations: int = 3
    max_concurrent_research: int = 2

    pdf_dir: Path = ROOT_DIR / "data" / "pdfs"
    processed_dir: Path = ROOT_DIR / "data" / "processed"

    log_level: str = "INFO"
    log_dir: Path = ROOT_DIR / "logs"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
