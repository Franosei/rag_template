"""Application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split_csv(raw_value: str) -> list[str]:
    """Split a comma-delimited string into trimmed non-empty values."""

    return [item.strip() for item in raw_value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Clinical Trials Hybrid Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    host: str = "127.0.0.1"
    port: int = 8000

    data_dir: Path = PROJECT_ROOT / "data"
    log_level: str = "INFO"
    log_format: Literal["pretty", "json"] = "pretty"

    default_top_k: int = 8
    chunk_size: int = 1100
    chunk_overlap: int = 160
    max_answer_citations: int = 8
    max_answer_context_chunks: int = 8
    max_answer_context_chars: int = 1800

    max_upload_mb: int = 15
    max_remote_file_mb: int = 25
    max_manifest_items: int = 20

    allowed_local_roots: str = Field(default="./data,.")
    trusted_hosts: str = Field(default="localhost,127.0.0.1,testserver")
    cors_origins: str = Field(default="http://127.0.0.1:8000,http://localhost:8000")
    auto_bootstrap_sample_data: bool = True

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.1
    llm_timeout_seconds: float = 45.0
    semantic_retrieval_enabled: bool = True
    embedding_batch_size: int = 24

    def model_post_init(self, __context: object) -> None:
        """Create expected runtime folders after settings load."""

        for folder in (
            self.data_dir,
            self.inbound_dir,
            self.processed_dir,
            self.indices_dir,
            self.folder_registry_dir,
            self.runs_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, raw_path: str | Path) -> Path:
        """Resolve a user-provided path relative to the project root when needed."""

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def inbound_dir(self) -> Path:
        """Return the managed inbound document directory."""

        return self.data_dir / "inbound"

    @property
    def processed_dir(self) -> Path:
        """Return the processed document output directory."""

        return self.data_dir / "processed"

    @property
    def indices_dir(self) -> Path:
        """Return the search-index storage directory."""

        return self.data_dir / "indices"

    @property
    def folder_registry_dir(self) -> Path:
        """Return the folder policy registry directory."""

        return self.data_dir / "folder_registry"

    @property
    def runs_dir(self) -> Path:
        """Return the execution trace directory."""

        return self.data_dir / "runs"

    @property
    def semantic_cache_path(self) -> Path:
        """Return the semantic embedding cache file."""

        return self.indices_dir / "semantic_embeddings.json"

    @property
    def allowed_local_root_paths(self) -> list[Path]:
        """Return normalized safe roots for local path ingestion."""

        roots = _split_csv(self.allowed_local_roots)
        if not roots:
            roots = [str(self.data_dir), str(PROJECT_ROOT)]
        return [self.resolve_path(root) for root in roots]

    @property
    def trusted_host_list(self) -> list[str]:
        """Return configured trusted hosts."""

        return _split_csv(self.trusted_hosts) or ["localhost", "127.0.0.1", "testserver"]

    @property
    def cors_origin_list(self) -> list[str]:
        """Return configured CORS origins."""

        return _split_csv(self.cors_origins)

    @property
    def llm_enabled(self) -> bool:
        """Indicate whether a remote LLM can be used."""

        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()


settings = get_settings()
