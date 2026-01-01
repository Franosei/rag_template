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

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- OpenAI API key or compatible LLM endpoint

### Installation
```bash
# Clone the repository
git clone https://github.com/Franosei/rag_template.git
cd graph-rag-multiagent-template

# Install dependencies (using uv)
uv sync

# Or using poetry
poetry install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys and settings
```

### Configuration

1. **Set up environment variables** in `.env`:
```
   OPENAI_API_KEY=your_key_here
   LOG_LEVEL=INFO
   VECTOR_STORE_TYPE=local  # or 'azure', 'aws', 'gcp'
```

2. **Customize configs** in `configs/`:
   - `app.yaml`: Core application settings
   - `prompts/`: Agent system prompts
   - `routing/`: Folder authority and routing rules

### Usage

#### Ingest Documents
```bash
# Drop folders into data/inbound/
cp -r /path/to/your/docs data/inbound/my_folder

# Run ingestion
./scripts/ingest_folder.sh my_folder
```

This will:
1. Profile the folder and generate a policy
2. Extract and chunk all documents
3. Build vector and keyword indices
4. Store metadata in the folder registry

#### Query the System
```bash
# CLI mode
python -m src.app.main query "What are the key findings in the Q4 report?"

# API mode (optional)
uvicorn src.api.server:app --reload
# Then POST to http://localhost:8000/api/query
```

#### Run with Docker
```bash
docker-compose up
```

## Project Structure

- **`src/agents/`**: Individual AI agents (profiler, retriever, verifier, etc.)
- **`src/core/`**: Orchestration engine, graph builder, policies, safety guardrails
- **`src/ingestion/`**: Document extraction, chunking, and indexing pipelines
- **`src/retrieval/`**: Hybrid search with vector, keyword, and metadata filtering
- **`src/api/`**: Optional FastAPI server for HTTP access
- **`configs/`**: YAML/JSON configs for prompts, schemas, and routing rules
- **`data/`**: Local storage for documents, indices, and execution traces (gitignored)
- **`tests/`**: Unit and integration tests with sample fixtures

## Use Cases

- **Enterprise Knowledge Bases**: Search across diverse internal documents with full audit trails
- **Regulatory Compliance**: Answer questions with verified citations to source documents
- **Research Assistants**: Synthesize findings from academic papers, patents, or reports
- **Customer Support**: Query product manuals, troubleshooting guides, and FAQs
- **Legal Discovery**: Analyze contracts, agreements, and case files with provenance

## Key Concepts

### Folder Policies
Each document folder gets a generated policy defining:
- Document types and schemas
- Expected entities and relationships
- Retrieval strategies (dense vs. sparse)
- Access and authority levels
- Custom metadata for routing

### Graph Traces
Every query execution builds a directed graph capturing:
- Which agents ran and in what order
- What sources were retrieved
- What reasoning steps were taken
- Which claims were verified
- Final answer composition

Graphs are stored in `data/runs/` for debugging and compliance.

### Authority-Based Routing
The system respects document hierarchy:
- Newer versions override older ones
- Official sources take precedence over drafts
- Scope selection considers user permissions (if configured)

## Development
```bash
# Run tests
pytest tests/

# Lint and format
./scripts/lint_format.sh

# Local development setup
./scripts/dev_setup.sh
```

## Deployment

See `docs/deployment_notes.md` for:
- Cloud vector store setup (Pinecone, Weaviate, Azure AI Search)
- Kubernetes deployment manifests
- Monitoring and observability integration
- Scaling considerations for production

## Roadmap

- [ ] Multi-language support for non-English documents
- [ ] Incremental indexing (update without full reprocessing)
- [ ] Real-time folder watching (inotify/fsevents)
- [ ] Graph visualization UI for execution traces
- [ ] Fine-tuned embeddings for domain-specific retrieval
- [ ] Integration with LangSmith/Weights & Biases for tracing

## Contributing

Contributions welcome! Please see:
- `.github/ISSUE_TEMPLATE/` for bug reports and feature requests
- `CONTRIBUTING.md` for development guidelines
- `LICENSE` for terms

## License

<https://unlicense.org>

## Acknowledgments

Built on principles from:
- Graph RAG (Microsoft Research)
- ReAct and multi-agent orchestration patterns
- Modern RAG best practices (hybrid search, reranking, verification)

---
