from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from pathlib import Path
from typing import Literal

class Settings(BaseSettings):
    """Global application settings loaded from .env and configs/"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 32000

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Storage
    vector_store_type: Literal["local", "pinecone", "weaviate"] = "local"
    data_dir: Path = Path("./data")
    
    # Retrieval
    top_k_retrieval: int = 20
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "pretty"] = "pretty"
    
    @validator("data_dir")
    def create_data_dirs(cls, v: Path) -> Path:
        """Ensure all data subdirectories exist"""
        (v / "inbound").mkdir(parents=True, exist_ok=True)
        (v / "processed").mkdir(parents=True, exist_ok=True)
        (v / "indices").mkdir(parents=True, exist_ok=True)
        (v / "folder_registry").mkdir(parents=True, exist_ok=True)
        (v / "runs").mkdir(parents=True, exist_ok=True)
        return v

# Singleton instance
settings = Settings()