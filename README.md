# Graph RAG Multi-Agent Template

A production-ready template for building intelligent document processing systems that combine Graph-based Retrieval-Augmented Generation (RAG) with multi-agent orchestration.

## Overview

This project provides a comprehensive framework for ingesting heterogeneous document collections, extracting structured knowledge, and answering complex queries through coordinated AI agents. Unlike traditional RAG systems, it maintains full provenance through graph traces and adapts retrieval strategies based on folder-specific policies.

## Key Features

### Intelligent Document Onboarding
- **Folder Policy System**: Automatically profiles document collections and generates metadata, access rules, and retrieval strategies per folder
- **Multi-format Support**: PDF, DOCX, XLSX, images, and tables with specialized extractors
- **Automated Processing**: Watch folders for new documents and trigger ingestion pipelines

### Multi-Agent Architecture
- **Specialized Agents**: Dedicated agents for scope selection, retrieval, table analysis, vision, reasoning, verification, and writing
- **Coordinated Execution**: Graph-based orchestration tracks agent interactions and decision flows
- **Flexible Composition**: Easily add, remove, or customize agents for your use case

### Hybrid Retrieval
- **Multi-strategy Search**: Combines vector similarity, keyword matching, and metadata filtering
- **Context-Aware**: Leverages folder policies to route queries to relevant document scopes
- **Authority-Based Routing**: Respects document hierarchy and access rules

### Full Observability
- **Execution Traces**: Every query generates a complete graph of agent actions, reasoning steps, and source citations
- **Claim Verification**: Built-in verifier agent validates answers against source documents
- **Export & Analysis**: Trace graphs exportable to JSON for debugging and auditing

### Production-Ready Safety
- **Unknown by Default**: System refuses to speculate when information isn't in the documents
- **Citation Requirements**: All claims must be traceable to specific sources
- **PII Redaction**: Optional guardrails for sensitive information
- **Guardrail Extensibility**: Easy to add custom safety checks

## Architecture Folder Directory

```
graph-rag-multiagent-template/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .env.example
├─ pyproject.toml
├─ uv.lock / poetry.lock (choose one)
├─ docker/
│  ├─ Dockerfile
│  ├─ docker-compose.yml
│  └─ entrypoint.sh
├─ scripts/
│  ├─ dev_setup.sh
│  ├─ run_local.sh
│  ├─ ingest_folder.sh
│  └─ lint_format.sh
├─ configs/
│  ├─ app.yaml
│  ├─ logging.yaml
│  ├─ folder_policy_schema.json
│  ├─ graph_schema.json
│  ├─ prompts/
│  │  ├─ folder_profiler.md
│  │  ├─ query_planner.md
│  │  ├─ verifier.md
│  │  └─ writer.md
│  └─ routing/
│     ├─ default_routes.yaml
│     └─ authority_rules.yaml
├─ data/                      # local-only, not committed (ignored)
│  ├─ inbound/                # folders dropped here to be onboarded
│  │  ├─ folder_001/
│  │  └─ folder_002/
│  ├─ processed/              # normalized extracts (chunks/tables/images)
│  ├─ indices/                # vector + keyword indices (local dev)
│  ├─ folder_registry/        # generated folder policies
│  └─ runs/                   # per-request traces + outputs
├─ src/
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ main.py              # CLI entry OR API entry wrapper
│  │  ├─ settings.py          # env + config loading
│  │  ├─ logging_config.py
│  │  └─ dependencies.py      # shared deps (llm client, stores)
│  ├─ api/                    # optional API layer (FastAPI)
│  │  ├─ __init__.py
│  │  ├─ server.py
│  │  ├─ routes/
│  │  │  ├─ health.py
│  │  │  ├─ ingest.py
│  │  │  └─ query.py
│  │  └─ schemas/
│  │     ├─ ingest.py
│  │     └─ query.py
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ orchestration/
│  │  │  ├─ graph_runner.py   # executes multi-agent flow + builds trace graph
│  │  │  ├─ state.py          # shared run state object
│  │  │  └─ events.py         # trace events & serialization
│  │  ├─ graph/
│  │  │  ├─ schema.py         # node/edge types + validation
│  │  │  ├─ builder.py        # add nodes/edges consistently
│  │  │  └─ exporters.py      # JSON export, claim ledger, etc.
│  │  ├─ policies/
│  │  │  ├─ folder_policy.py  # folder policy model + validator
│  │  │  ├─ registry.py       # read/write registry of folders
│  │  │  └─ routing.py        # scope selection + authority rules
│  │  └─ safety/
│  │     ├─ pii.py            # redaction hooks if needed
│  │     └─ guardrails.py     # “unknown by default”, citations required, etc.
│  ├─ agents/
│  │  ├─ __init__.py
│  │  ├─ base.py              # Agent interface
│  │  ├─ folder_profiler.py   # generates Folder Policy (JSON)
│  │  ├─ scope_selector.py
│  │  ├─ retriever.py
│  │  ├─ table_agent.py
│  │  ├─ vision_agent.py
│  │  ├─ reasoner.py
│  │  ├─ verifier.py
│  │  └─ writer.py
│  ├─ ingestion/
│  │  ├─ __init__.py
│  │  ├─ folder_watcher.py    # detects new folders (polling for local dev)
│  │  ├─ pipeline.py          # end-to-end onboarding
│  │  ├─ extractors/
│  │  │  ├─ pdf.py
│  │  │  ├─ docx.py
│  │  │  ├─ xlsx.py
│  │  │  ├─ images.py
│  │  │  └─ common.py
│  │  └─ chunking/
│  │     ├─ chunker.py
│  │     └─ table_normalizer.py
│  ├─ retrieval/
│  │  ├─ __init__.py
│  │  ├─ hybrid.py            # keyword + vector + metadata filters
│  │  ├─ embeddings.py
│  │  ├─ stores/
│  │  │  ├─ vector_store.py   # interface
│  │  │  ├─ local_faiss.py    # local dev implementation
│  │  │  └─ cloud_adapter.py  # placeholder adapters (Azure/AWS/GCP)
│  │  └─ ranking.py
│  ├─ llm/
│  │  ├─ __init__.py
│  │  ├─ client.py            # OpenAI v1 client wrapper
│  │  ├─ prompts.py           # loads prompts from configs/
│  │  └─ json_mode.py         # strict JSON output utilities
│  └─ utils/
│     ├─ __init__.py
│     ├─ fileio.py
│     ├─ hashing.py
│     ├─ ids.py
│     ├─ timing.py
│     └─ validation.py
├─ tests/
│  ├─ unit/
│  │  ├─ test_folder_policy.py
│  │  ├─ test_graph_schema.py
│  │  └─ test_retrieval.py
│  ├─ integration/
│  │  ├─ test_onboarding_pipeline.py
│  │  └─ test_query_flow.py
│  └─ fixtures/
│     └─ sample_docs/
├─ docs/
│  ├─ architecture.md
│  ├─ graph_spec.md
│  ├─ folder_policy_spec.md
│  └─ deployment_notes.md
└─ .github/
   ├─ workflows/
   │  ├─ ci.yml
   │  └─ security.yml
   └─ ISSUE_TEMPLATE/
      ├─ bug_report.md
      └─ feature_request.md
```

### Architecture
```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Folder Profiler    │  Generates policy, schema, metadata
│  (Agent)            │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Ingestion Pipeline │  Extract → Chunk → Embed → Index
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Vector + Keyword   │
│  Indices            │
└──────┬──────────────┘
       │
       ▼
    [Query]
       │
       ▼
┌─────────────────────┐
│  Multi-Agent Flow   │
│                     │
│  1. Scope Selector  │  Which folders are relevant?
│  2. Retriever       │  Fetch candidate chunks
│  3. Table Agent     │  Parse structured data
│  4. Vision Agent    │  Analyze images/charts
│  5. Reasoner        │  Synthesize information
│  6. Verifier        │  Validate claims
│  7. Writer          │  Format final answer
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Answer + Graph     │  Response with full provenance
└─────────────────────┘
```

## Quick Start

