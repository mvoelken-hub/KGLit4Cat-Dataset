---
name: simone
description: "Use when working with the SIMONE (Semantic Inference Module for Ontology-driven Node Extraction) project. Covers the full-stack architecture: FastAPI backend with Pydantic AI agents, Neo4j graph database, Ollama LLM integration, React+Vite frontend, and Docker Compose infrastructure. Use for: adding API endpoints, modifying extraction workflows, frontend UI changes, Neo4j schema updates, or Docker configuration."
---

# SIMONE Project Knowledge

## Architecture Overview

SIMONE is a metadata extraction pipeline that takes dataset archives (ZIP files), chunks them, extracts initial context via LLM agents, generates a DCAT-AP profile draft, and patches it against vocabulary-backed knowledge graphs.

### Stack
- **Backend**: Python 3.11+, FastAPI, Pydantic AI, uv package manager
- **Database**: Neo4j (with APOC plugin) for semantic graph storage
- **LLM**: Ollama (local) with configurable embed/chat models
- **Frontend**: React 19, TypeScript, Vite dev server, vanilla CSS (no framework)
- **Infra**: Docker Compose (production + dev variants)

## Project Structure

```
backend/
  app/
    api/v1/           # FastAPI routers (extraction, profiles, datasources, semantic, system)
      schemas/        # Pydantic request/response models for all v1 endpoints
    bootstrap.py      # Startup orchestration: pulls Ollama models, imports initial vocabularies
    cli.py            # Typer CLI for Docker Compose management, backups, health checks
    core/             # Config, logging, task registry, initial vocabularies
    dependencies.py   # Manual DI container: instantiates singleton repositories and services
    domain/           # Business logic: datasources, extraction agents, profiles, semantics
    neo4j/            # Driver, indexes, types
    ollama/           # Client wrapper
    repositories/     # Protocol classes for persistence
    services/         # Service layer orchestrating repositories and agents
  infra/              # Filesystem and Neo4j implementations of repository protocols
  tests/              # pytest suite
frontend/
  src/
    api/              # API client functions (client, datasources, extraction, profiles, semantic, types)
    components/       # Reusable UI components (ChunkingDialog, JsonEditor, VocabularyPanel)
    App.tsx           # Main application shell
    styles.css        # All styles in one file
  vite.config.ts      # Vite config with /api/v1 proxy to localhost:8000
```

## Key Backend Patterns

### Extraction Workflow
1. `POST /extraction/initial-context` — extracts high-level context from data package
2. `POST /extraction/initial-draft` — generates DCAT-AP draft from context + profile schema
3. `POST /extraction/patch-draft` — background task that patches draft against content chunks

### Repository Pattern
All persistence goes through Protocol classes in `app/repositories/`, implemented in `infra/`:
- `ExtractionOutputRepository` — saves/loads JSON artifacts per workflow_id
- `DataSourceBlobRepository` — stores uploaded ZIP files
- `ProfileRepository` — stores registered DCAT-AP profiles
- `SemanticGraphRepository` — Neo4j graph operations (vocabularies, embeddings, queries)

### Dependency Injection
`backend/app/dependencies.py` acts as a manual DI container. It instantiates repository and service singletons at module import time (e.g., `datasource_blob_repository = FileSystemDataSourceBlobRepository(...)`) and provides getter functions used by FastAPI `Depends`.

### Task Registry
Long-running operations (chunking, patching) use `TaskRegistry` with `TaskStatus` enum:
`running | completed | cancelled | crashed | unknown`

## Key Frontend Patterns

### API Client
`frontend/src/api/client.ts` defines:
- `apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1'`
- `readJson<T>()` — parses JSON, throws `ApiError` on non-ok responses
- `buildQuery()` — helper for URLSearchParams

Additional API modules:
- `frontend/src/api/semantic.ts` — vocabulary import, embedding checks, vocabulary queries
- `frontend/src/api/types.ts` — shared TypeScript types (TaskStatus, ChunkResponse, VocabQueryResult, etc.)

### State Management
Pure React hooks in `App.tsx`. No external state library.
Key states: `packages`, `profiles`, `selectedPackageId`, `selectedProfile`, `context`, `draft`, `busy`, `message`

### Components
- `JsonEditor.tsx` — Tree sidebar showing only containers (objects/arrays) with child counts; editable panel with form fields for strings, numbers, booleans; array items render as clickable cards; raw JSON toggle for reference; live updates via `onChange` callback
- `ChunkingDialog.tsx` — Modal for configuring dataset chunking parameters (buffer window size, semantic threshold, text quality filters, protected lines)
- `VocabularyPanel.tsx` — Modal for browsing/importing controlled vocabularies, checking embedding status, and running vocabulary queries

### CSS Architecture
All styles in `frontend/src/styles.css`. Uses CSS custom properties:
```css
:root {
  --paper: #f4f0e8;
  --ink: #17140f;
  --muted: #6e675d;
  --line: #d8d0c1;
  --panel: rgba(255, 252, 246, 0.76);
  --accent: #8d3f22;
  --accent-dark: #582614;
}
```

## Development Setup

### Prerequisites
- Node.js (v24+ recommended) — install via `winget install OpenJS.NodeJS.LTS`
- Python 3.11+ with `uv` package manager
- Docker Desktop

### Start Backend (dev)
```powershell
cd backend
# Use the project .env file (the CLI creates it from .env.example on first use)
uv run --env-file ../.env uvicorn app.main:fastapi_app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend (dev)
```powershell
cd frontend
npm install
npx vite --port 3000
```
The Vite dev server proxies `/api/v1` to `http://127.0.0.1:8000`.

#### Troubleshooting Frontend Startup
- **Node.js not on PATH**: If `npx` or `npm` is not found, ensure Node.js is installed and on your PATH, or run Vite directly:
  ```powershell
  cd frontend
  & "C:\Program Files\nodejs\node.exe" "node_modules\vite\bin\vite.js" --port 3000
  ```
- **PowerShell Execution Policy**: If `npm`/`npx` fails with a PSSecurityException, PowerShell script execution is disabled. Either run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` (requires admin), or use the direct `node.exe` command above.

### Start Full Stack (CLI)
```powershell
# Production mode (API + frontend in Docker; Neo4j/Ollama in Docker unless .env points to remote hosts)
simone up

# Dev mode (API locally; Neo4j/Ollama in Docker unless .env points to remote hosts)
simone dev

# Dev mode with Docker frontend (no local npm needed)
simone dev --no-npm

# Infrastructure mode (only Neo4j/Ollama, no API/frontend — for remote host setups)
simone host

# Start only specific services
simone host --neo4j
simone host --ollama
```

The CLI creates `.env` from `.env.example` on first use and sets `APP_ENV` automatically (`production` for `up` and `host`, `development` for `dev`). If `NEO4J_HOSTNAME` or `OLLAMA_HOSTNAME` in `.env` point to a remote host, the CLI skips the corresponding local Docker service.

## Common Tasks

### Adding a New API Endpoint
1. Add request/response schemas in `backend/app/api/v1/schemas/<domain>.py`
2. Add route handler in `backend/app/api/v1/<domain>.py`
3. Add service method in `backend/app/services/<domain>_service.py`
4. Wire up dependency injection in `backend/app/dependencies.py` (instantiate repository/service singletons)
5. Add frontend API function in `frontend/src/api/<domain>.ts`
6. Update `frontend/src/App.tsx` to use it

### Modifying the JsonEditor
- Tree rendering: `TreeNode` component in `JsonEditor.tsx`
- Field editors: `ValueEditor` component
- Styles: `.json-editor-*` classes in `styles.css`

### Adding a New Step to the Workflow
1. Add article in `App.tsx` workflow section
2. Add handler function (e.g., `onNewStep()`)
3. Add `BusyKey` variant if needed
4. Add API function in `frontend/src/api/`

### Working with Vocabularies and Semantic Graph
- Backend domain logic: `backend/app/domain/semantics/` (controlled_vocabularies.py, ontologies.py, rdf.py, vocab_queries.py)
- Backend service: `backend/app/services/semantic_service.py`
- Backend repository: `infra/neo4j_semantic_graph_repository.py`
- Frontend API: `frontend/src/api/semantic.ts`
- Frontend UI: `frontend/src/components/VocabularyPanel.tsx`
- API endpoints: `backend/app/api/v1/semantic.py` (prefix `/semantic`)
- Initial vocabularies loaded at startup are defined in `backend/app/core/initial_vocabs.py`

## Environment Files
- `.env.example` — template with shared defaults (hostnames, ports, model settings). No `APP_ENV` — the CLI sets it per command.
- `.env` — runtime file created from `.env.example` by the CLI. Gitignored.

Host behavior: if `NEO4J_HOSTNAME` or `OLLAMA_HOSTNAME` is empty, `localhost`, or `127.0.0.1`, the CLI starts the corresponding Docker service. Remote host values are used directly and the local service is skipped.

## Important Notes
- The frontend uses `"latest"` for all npm deps — `package-lock.json` is gitignored
- Backend uses `uv.lock` (not committed, per Python library convention)
- Neo4j password is only applied on first data directory initialization
- Ollama models are pulled on first startup in production mode unless `SKIP_MODEL_PULL=true`
