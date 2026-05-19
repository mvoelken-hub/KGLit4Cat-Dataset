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
    core/             # Config, logging, task registry
    domain/           # Business logic: datasources, extraction agents, profiles, semantics
    neo4j/            # Driver, indexes, types
    ollama/           # Client wrapper
    repositories/     # Protocols for persistence
    services/         # Service layer orchestrating repositories and agents
  infra/              # Filesystem implementations of repositories
  tests/              # pytest suite
frontend/
  src/
    api/              # API client functions (datasources, extraction, profiles, types)
    components/         # Reusable UI components (JsonEditor)
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
- `SemanticGraphRepository` — Neo4j graph operations

### Task Registry
Long-running operations (chunking, patching) use `TaskRegistry` with `TaskStatus` enum:
`running | completed | cancelled | crashed | unknown`

## Key Frontend Patterns

### API Client
`frontend/src/api/client.ts` defines:
- `apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1'`
- `readJson<T>()` — parses JSON, throws `ApiError` on non-ok responses
- `buildQuery()` — helper for URLSearchParams

### State Management
Pure React hooks in `App.tsx`. No external state library.
Key states: `packages`, `profiles`, `selectedPackageId`, `selectedProfile`, `context`, `draft`, `busy`, `message`

### JsonEditor Component
`frontend/src/components/JsonEditor.tsx`
- Tree sidebar showing only containers (objects/arrays) with child counts
- Editable panel with form fields for strings, numbers, booleans
- Array items render as clickable cards
- Raw JSON toggle for reference
- Live updates via `onChange` callback

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
uv run --env-file ../.env.development uvicorn app.main:fastapi_app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend (dev)
```powershell
cd frontend
npm install
npx vite --port 3000
```
The Vite dev server proxies `/api/v1` to `http://127.0.0.1:8000`.

### Start Full Stack (Docker)
```powershell
# Production mode
docker compose --env-file .env.production up -d --build

# Dev mode (Neo4j + Ollama in Docker, API + Frontend local)
docker compose --env-file .env.development -f docker-compose.dev.yml up -d
```

## Common Tasks

### Adding a New API Endpoint
1. Add route handler in `backend/app/api/v1/<domain>.py`
2. Add request/response schemas in `backend/app/api/v1/schemas/`
3. Add service method in `backend/app/services/<domain>_service.py`
4. Wire up dependency injection in `backend/app/dependencies.py`
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

## Environment Files
- `.env.development.example` — local dev config (Neo4j on localhost, Ollama on localhost)
- `.env.production.example` — Docker config (Neo4j/ Ollama via service names)

## Important Notes
- The frontend uses `"latest"` for all npm deps — `package-lock.json` is gitignored
- Backend uses `uv.lock` (not committed, per Python library convention)
- Neo4j password is only applied on first data directory initialization
- Ollama models are pulled on first startup unless `SKIP_MODEL_PULL=true`
