"""Request and response schemas for the FastAPI layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SourceRegistrationRequest(BaseModel):
    """Payload for registering a new data source."""

    source_type: Literal["seed_dataset", "local_path", "remote_url", "api_manifest", "inline_upload"]
    location: str | None = None
    folder_name: str | None = None
    filename: str | None = None
    content_base64: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "SourceRegistrationRequest":
        """Validate the required fields for each source type."""

        if self.source_type in {"local_path", "remote_url", "api_manifest"} and not self.location:
            raise ValueError("location is required for the selected source type.")
        if self.source_type == "inline_upload" and (not self.filename or not self.content_base64):
            raise ValueError("filename and content_base64 are required for inline uploads.")
        return self


class QueryRequest(BaseModel):
    """Payload for a user query."""

    question: str
    folder_ids: list[str] = Field(default_factory=list)
    top_k: int | None = None
