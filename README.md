# Clinical Trials Hybrid Agent

Citation-first semantic document intelligence workspace for clinical-trials and regulatory documents.

This repository provides a FastAPI backend and a professional browser UI for teams that need to ingest complex documents, search them by meaning, and generate grounded answers with evidence that can be inspected directly.

## Overview

Clinical and regulatory teams often have the same problem:

- documents live in different places
- answers need traceable evidence
- keyword-only search is not enough
- non-technical users still need a usable interface
- developers need an API and a clean backend they can extend

This project solves that by combining:

- semantic retrieval with cached embeddings
- hybrid lexical retrieval for exact terminology
- graph-based retrieval expansion
- citation-aware answer synthesis
- a dashboard that exposes the evidence behind each answer

## Why This Repo Exists

This repo is useful when you want to build or run a system that:

- works on professional document collections
- gives answers with citations instead of unsupported summaries
- supports clinical-trials, regulatory, medical, or research-style PDFs and office documents
- can be used both by developers and by non-technical teammates
- is ready to run locally with FastAPI and easy to integrate into another application

## Why Someone Would Clone This Repo

Clone this repository if you need:

- a ready-to-run RAG application with a usable frontend
- a starting point for a clinical or regulatory document assistant
- a FastAPI backend that exposes ingestion and query endpoints
- a semantic retrieval workflow with local caching
- a reference implementation for citation-first document QA

## Who It Is For

### Developers

This repo is for developers who want to:

- build internal AI search tools
- integrate document-grounded answers into another product
- use FastAPI as the backend for a retrieval system
- extend ingestion, chunking, retrieval, or answer orchestration
- prototype a secure local-first document AI workflow

### Non-Technical Users

This repo is also for teams where the end user is not a developer, such as:

- clinical operations teams
- medical affairs teams
- regulatory teams
- research analysts
- program managers reviewing evidence

Those users can work through the browser UI without writing code.

## What The System Does

- Ingests documents from:
  - local paths inside approved safe roots
  - remote URLs
  - API manifests that return document URLs
  - direct browser uploads
- Extracts and chunks text from supported files
- Builds a retrieval layer using:
  - semantic retrieval
  - hybrid lexical retrieval
  - graph retrieval
- Fuses the retrieved evidence into a final ranked set
- Produces grounded answers with citations
- Opens citation details in a right-side evidence drawer with:
  - document name
  - page number
  - retrieval mode
  - matched terms
  - supporting passage
  - source path

## Main Features

- FastAPI API for ingestion and question answering
- Professional web dashboard for document search and evidence review
- Semantic retrieval with cached OpenAI embeddings
- Hybrid lexical retrieval for exact terminology and phrase grounding
- Graph-based concept expansion
- Citation drawer for human-readable evidence inspection
- Retrieval diagnostics and run trace visibility
- Controlled local-path access and size-limited remote ingestion

## Retrieval Architecture

The system does not rely on keyword overlap alone.

It uses three retrieval layers:

1. Semantic retrieval
   Ranks chunks by meaning using embeddings.
2. Hybrid lexical retrieval
   Adds exact-term grounding using TF-IDF and BM25-style scoring.
3. Graph retrieval
   Expands related concepts and nearby evidence.

These channels are fused before answer synthesis.

## Supported Data Sources

- `local_path`
- `remote_url`
- `api_manifest`
- `inline_upload`
- `seed_dataset`

## Supported File Types

- `.pdf`
- `.docx`
- `.xlsx`
- `.xls`
- `.csv`
- `.txt`
- `.md`
- `.rst`

## Quick Start

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd rag_template
```

### 2. Create a virtual environment

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
py -3 -m pip install -e .
```

### 4. Create your environment file

```powershell
Copy-Item .env.example .env
```

Set at least:

- `OPENAI_API_KEY`

### 5. Start the app

```powershell
py -3 -m uvicorn src.api.server:app --reload
```

### 6. Open the dashboard

```text
http://127.0.0.1:8000
```

By default, the bundled clinical-trials sample corpus is bootstrapped on startup.

## First-Run Notes

On first startup with a valid API key:

- the bundled sample corpus is ingested
- semantic embeddings are created and cached locally
- startup can take longer than later runs

Subsequent runs are faster because cached embeddings are reused from `data/indices/semantic_embeddings.json`.

## How Developers Can Use It

### Option 1: Run it as a complete application

Use the existing backend and frontend as a full document intelligence workspace.

This is useful for:

- internal pilots
- team knowledge workspaces
- document QA portals
- evidence review tools

### Option 2: Integrate the API into another app

You can call these endpoints from your own frontend or service:

- `GET /api/health`
- `GET /api/sources`
- `POST /api/sources`
- `POST /api/query`

Example query payload:

```json
{
  "question": "What do the ICH E9 and E9(R1) materials say about estimands?",
  "folder_ids": [],
  "top_k": 8
}
```

### Option 3: Use the Python service layer directly

The main orchestration entry point is `src/services/knowledge_base.py`.

You can instantiate `KnowledgeBaseService` and call:

- `bootstrap()`
- `ingest_local_path()`
- `ingest_remote_url()`
- `ingest_api_manifest()`
- `ingest_inline_upload()`
- `query()`

### What Developers Get Back

The query workflow returns:

- answer text
- ranked citations
- retrieval diagnostics
- selected folder metadata
- retrieval hits
- run trace

That makes it suitable for:

- internal copilots
- QA assistants
- review systems
- regulatory search portals
- research evidence tools

## How Non-Technical Users Can Use It

### 1. Open the dashboard

Use `http://127.0.0.1:8000`.

### 2. Add documents

Choose one of:

- local path
- remote URL
- API manifest
- upload from browser

### 3. Select the collections you want

Use the Indexed Collections panel to limit the search scope.

### 4. Ask a question in plain language

Examples:

- "What is an estimand?"
- "How should intercurrent events be handled?"
- "What do these documents say about trial objectives?"

### 5. Open the evidence

Click the evidence references under the answer or in the citations panel to open the evidence drawer.

## Environment Variables

Important settings in `.env`:

- `OPENAI_API_KEY`
  Required for semantic retrieval and remote answer synthesis.
- `OPENAI_BASE_URL`
  Defaults to `https://api.openai.com/v1`.
- `OPENAI_MODEL`
  Chat model used for answer synthesis.
- `OPENAI_EMBEDDING_MODEL`
  Embedding model used for semantic retrieval.
- `SEMANTIC_RETRIEVAL_ENABLED`
  Enables semantic retrieval when embeddings are available.
- `DEFAULT_TOP_K`
  Number of final retrieval hits requested from the retrieval pipeline.
- `MAX_ANSWER_CITATIONS`
  Maximum number of citations returned to the UI.
- `MAX_ANSWER_CONTEXT_CHUNKS`
  Maximum number of retrieved chunks passed into answer synthesis.
- `MAX_ANSWER_CONTEXT_CHARS`
  Per-chunk character budget for answer synthesis context.
- `ALLOWED_LOCAL_ROOTS`
  Safe local directories allowed for `local_path` ingestion.
- `AUTO_BOOTSTRAP_SAMPLE_DATA`
  Controls whether the sample dataset is indexed on startup.

## Example Source Registration Payloads

Local path:

```json
{
  "source_type": "local_path",
  "location": "C:/safe-root/my-trial-docs",
  "folder_name": "phase3_docs"
}
```

Remote URL:

```json
{
  "source_type": "remote_url",
  "location": "https://example.org/protocol.pdf",
  "folder_name": "downloaded_protocol"
}
```

API manifest:

```json
{
  "source_type": "api_manifest",
  "location": "https://example.org/api/documents",
  "folder_name": "sponsor_feed"
}
```

Browser upload:

```json
{
  "source_type": "inline_upload",
  "filename": "protocol.pdf",
  "content_base64": "<base64 payload>"
}
```

## Project Layout

```text
src/
  api/             FastAPI routes and request schemas
  agents/          folder profiling logic
  app/             settings, logging, local entrypoint
  core/            policies, trace graph, orchestration state
  ingestion/       extraction, chunking, and processing pipeline
  llm/             OpenAI-compatible text and embedding client
  retrieval/       semantic, hybrid, graph, and tokenization modules
  services/        ingestion and query orchestration
  web/             static frontend assets
tests/             smoke tests
data/
  inbound/         managed source folders and bundled sample corpus
  processed/       generated chunk and document outputs
  indices/         cached semantic embeddings
  folder_registry/ indexed collection policies
  runs/            saved query traces
```

## Security And Operational Defaults

- Local path ingestion is restricted to configured safe roots.
- Unsupported file types are rejected early.
- Remote downloads are size-limited before and during transfer.
- Manifest ingestion is capped by `MAX_MANIFEST_ITEMS`.
- The app stores processed chunks, retrieval traces, and embedding cache locally.
- If the LLM is unavailable, the app falls back conservatively instead of inventing unsupported claims.

## Verification

The current implementation has been verified locally with:

- `py -3 -m compileall src`
- `py -3 -m pytest -q`
- end-to-end ingestion and query runs against the bundled clinical-trials corpus

## Notes For Production Teams

This repo is a strong local and internal starting point, but production teams will usually want to add:

- authentication and access control
- audit logging and observability
- persistent external storage
- managed vector infrastructure
- deployment automation
- governance and retention controls

## Summary

This repository is best understood as:

- a professional FastAPI RAG starter for clinical and regulatory documents
- a citation-first semantic document QA system
- a usable browser workspace for non-technical users
- a backend foundation developers can integrate into larger systems
