"""FastAPI server for the clinical-trials hybrid agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import QueryRequest, SourceRegistrationRequest
from src.app.logging_config import setup_logging
from src.app.settings import settings
from src.services.knowledge_base import KnowledgeBaseService

setup_logging(settings.log_level, settings.log_format)
service = KnowledgeBaseService()
web_root = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Run application startup tasks."""

    if settings.auto_bootstrap_sample_data:
        await service.bootstrap()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list or ["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_host_list)
app.mount("/assets", StaticFiles(directory=web_root), name="assets")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """Serve the single-page frontend."""

    return FileResponse(web_root / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    """Return application status for the UI."""

    return {"status": "ok", "app_name": settings.app_name, "stats": service.stats()}


@app.get("/api/sources")
async def list_sources() -> dict[str, object]:
    """Return registered source folders."""

    return {"sources": service.list_sources(), "stats": service.stats()}


@app.post("/api/sources")
async def register_source(request: SourceRegistrationRequest) -> dict[str, object]:
    """Register and ingest a source."""

    try:
        if request.source_type == "seed_dataset":
            result = await service.ingest_seed_dataset()
            return {"mode": "seed_dataset", "result": result, "stats": service.stats()}
        if request.source_type == "local_path":
            result = await service.ingest_local_path(request.location or "", request.folder_name)
        elif request.source_type == "remote_url":
            result = await service.ingest_remote_url(request.location or "", request.folder_name)
        elif request.source_type == "api_manifest":
            result = await service.ingest_api_manifest(request.location or "", request.folder_name)
        else:
            result = await service.ingest_inline_upload(
                filename=request.filename or "",
                content_base64=request.content_base64 or "",
                folder_name=request.folder_name,
            )
        return {"mode": request.source_type, "result": result.model_dump(mode="json"), "stats": service.stats()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/query")
async def query(request: QueryRequest) -> dict[str, object]:
    """Answer a question using the indexed corpus."""

    try:
        return await service.query(request.question, folder_ids=request.folder_ids or None, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
