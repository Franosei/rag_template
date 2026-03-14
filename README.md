# Clinical Trials Hybrid Agent

FastAPI-based hybrid retrieval application for clinical-trials and regulatory documents.

## What This Repo  Includes

- A FastAPI backend with health, source-ingestion, and query endpoints
- A browser UI for indexing sources and querying the corpus
- Deterministic folder profiling for clinical-trials-style document sets
- Local hybrid retrieval that combines TF-IDF and BM25-style keyword ranking
- Citation-aware answers with extractive fallback when no remote LLM is configured
- Safe source intake for:
  - local filesystem paths under configured safe roots
  - remote `http` and `https` documents
  - API manifests that return document URLs
  - browser uploads

The bundled sample corpus under `data/inbound/` contains FDA, EMA, ICH, publication, and R package PDFs related to clinical trials and design of experiments.

## Project Layout

```text
src/
  api/             FastAPI routes and request schemas
  app/             runtime settings, logging, local entrypoint
  agents/          folder profiling logic
  core/            policies, trace graph, orchestration state
  ingestion/       extractors, chunking, processing pipeline
  retrieval/       local hybrid retriever
  services/        ingestion/query orchestration service
  web/             static frontend assets
data/
  inbound/         source folders and sample clinical-trials documents
  processed/       generated documents/chunks per folder
  folder_registry/ generated folder policies
  runs/            saved query traces
```

## Quick Start

1. Create a virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
py -3 -m pip install -e .
```

2. Copy the environment template:

```powershell
Copy-Item .env.example .env
```

3. Start the app:

```powershell
py -3 -m uvicorn src.api.server:app --reload
```

4. Open:

```text
http://127.0.0.1:8000
```

By default the server bootstraps the sample clinical-trials corpus on startup.

## Environment Notes

Important variables in `.env`:

- `ALLOWED_LOCAL_ROOTS`
  - Comma-separated directories the server is allowed to read from for `local_path` ingestion
- `OPENAI_API_KEY`
  - Optional. If set, the app attempts remote answer synthesis through an OpenAI-compatible chat completions endpoint
- `OPENAI_BASE_URL`
  - Defaults to `https://api.openai.com/v1`
- `AUTO_BOOTSTRAP_SAMPLE_DATA`
  - When `true`, the bundled sample data is indexed on startup

## API Endpoints

- `GET /api/health`
- `GET /api/sources`
- `POST /api/sources`
- `POST /api/query`

Example source registration payloads:

```json
{
  "source_type": "local_path",
  "location": "C:/safe-root/my-trial-docs",
  "folder_name": "phase3_docs"
}
```

```json
{
  "source_type": "remote_url",
  "location": "https://example.org/protocol.pdf",
  "folder_name": "downloaded_protocol"
}
```

```json
{
  "source_type": "api_manifest",
  "location": "https://example.org/api/documents"
}
```

```json
{
  "source_type": "inline_upload",
  "filename": "protocol.pdf",
  "content_base64": "<base64 payload>"
}
```

Example query payload:

```json
{
  "question": "What do the ICH guidelines say about estimands?",
  "folder_ids": [],
  "top_k": 8
}
```

## Verification

The current codebase has been verified locally with:

- `py -3 -m compileall src`
- FastAPI `TestClient` requests against `/api/health` and `/api/query`
- An end-to-end bootstrap and query run against the bundled clinical-trials PDFs

## Security Defaults

- Local path ingestion is restricted to configured safe roots
- Remote files are size-limited before and during download
- Manifest ingestion is capped by `MAX_MANIFEST_ITEMS`
- Unsupported file types are rejected early
- The query flow is citation-aware and falls back conservatively when synthesis is unavailable
