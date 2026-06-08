# SIMONE Codebase And Runtime Map

Use this reference for codebase navigation, runtime operations, layer boundaries, artifacts, and test commands.

Primary local sources to re-check:

- `README.md`
- `docs/WORKFLOW.md`
- `backend/app`
- `frontend/src`
- `docker-compose.yml`
- `.env` and `.env.example`

## Repository Root

`C:\Users\simcl\Documents\GitHub\Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)`

Wrappers:

- Windows: `.\simone.bat`
- macOS/Linux: `./simone`

The wrapper delegates to `uv --directory backend run simone`, whose Typer app is `backend/app/cli.py`.

## Runtime Modes

- `.\simone.bat dev`: development mode, API locally with Uvicorn reload, frontend through npm/Vite when available, local Neo4j/Ollama containers when env hostnames are local.
- `.\simone.bat dev --no-npm`: Docker frontend mode.
- `.\simone.bat dev -fg`: attach local dev logs to current terminal.
- `.\simone.bat up`: production mode, API/frontend in Docker.
- `.\simone.bat host`: infrastructure-only Neo4j/Ollama.
- `.\simone.bat down`: stop containers, local dev API/frontend processes, and background vocabulary bootstrap jobs.
- `.\simone.bat status`: inspect stack status before launch/restart.

Do not use stale paths such as `.env.development` or `docker-compose.dev.yml`.

## Important Environment Fields

- `NEO4J_HOSTNAME` and `OLLAMA_HOSTNAME`: empty, `localhost`, `127.0.0.1`, or `::1` means local Docker services are managed by the CLI.
- `NEO4J_PORT`: default `7687`.
- `OLLAMA_PORT`: default `11433`.
- `FRONTEND_PORT`: default `3000`.
- `RUNTIME_DIR`: default `./.runtime`.
- `SKIP_MODEL_PULL` and `SKIP_INITIAL_VOCAB_IMPORT`: startup overrides.

Neo4j persists under `data/docker/neo4j/data`; initial password only applies when that directory is first initialized.

## URLs

- Frontend: `http://127.0.0.1:3000/`
- Backend docs: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/api/v1/health`
- Neo4j Browser: `http://127.0.0.1:7474/browser/` when local

## Runtime Artifacts

Local dev API runtime is normally `backend/.runtime` because Uvicorn runs from `backend`.

Production API container runtime volume is `data/docker/api/data`, mounted to `/app/.runtime`.

Extraction output artifacts are normally under:

`backend/.runtime/output/<data_package_id>`

Current files:

- `extraction_context.json`
- `extraction_result.json`
- `extraction_run_state.json`
- `extraction_warnings.json`
- `token_usage.json`

Avoid committing `.runtime`, `data/docker`, generated workflow files, or local service data unless the user explicitly asks.

## Architecture

Primary locations:

- `backend/app/main.py`: FastAPI app, lifespan startup/shutdown, CORS, router registration.
- `backend/app/bootstrap.py`: startup setup for Ollama, Neo4j, graph cleanup, initial vocabulary import.
- `backend/app/dependencies.py`: composition root for settings, clients, repositories, and services.
- `backend/app/cli.py`: Typer CLI behind wrappers.
- `backend/app/core`: settings, logging, task registry, initial vocabularies.
- `backend/app/api/v1`: HTTP routers.
- `backend/app/api/v1/schemas`: API request/response DTOs.
- `backend/app/services`: application orchestration.
- `backend/app/domain`: domain models, algorithms, prompts, schema/profile logic, RDF/vocabulary logic, datasource parsing.
- `backend/app/repositories`: service-facing repository protocols.
- `backend/infra`: filesystem and Neo4j repository implementations.
- `backend/app/neo4j`: Neo4j driver/index support.
- `backend/app/ollama`: Ollama client/completion/runtime helpers.
- `frontend/src/App.tsx`: main workflow UI.
- `frontend/src/api`: typed fetch wrappers.
- `frontend/src/components`: reusable UI controls.
- `docs/thesis`: LaTeX thesis.
- `docs/thesis/assets/agent-generated-assets`: thesis planning/working docs.

## Layer Boundaries

Keep dependencies flowing inward:

```text
api -> services -> domain
api -> schemas
services -> repository protocols -> infra implementations
services -> core/ollama/neo4j clients as needed
```

API routers should parse requests, call services through dependency injection, translate expected exceptions into `HTTPException`, and return API schemas.

Services coordinate use cases across domain functions, repositories, Ollama, Neo4j, settings, and `TaskRegistry`.

Domain code should not import API, services, infra, FastAPI, or concrete clients.

When services need new persistence behavior, add it to the matching protocol in `backend/app/repositories` first, then implement it in `backend/infra`.

Wire new services/repositories in `backend/app/dependencies.py`.

## Common Lookup Paths

- Startup/CLI issue: `backend/app/cli.py`, `docker-compose.yml`, `.env`, `.env.example`, `backend/app/core/config.py`, `backend/app/bootstrap.py`, `backend/app/main.py`.
- Health/settings issue: `backend/app/api/v1/system.py`, `backend/app/core/config.py`, `backend/app/ollama/runtime.py`.
- API behavior: `backend/app/api/v1/<area>.py` -> `backend/app/services/<area>_service.py` -> domain/repository calls.
- Extraction behavior: `backend/app/services/extraction_service.py`, `backend/app/domain/extraction/extraction_context.py`, `file_ranking.py`, `profile_projection.py`, `vocabulary.py`, `workflow.py`.
- Datasource upload/chunking: `backend/app/services/datasource_service.py`, `backend/app/domain/datasources`, datasource repositories.
- Profile registration/validation/export: `backend/app/services/profile_service.py`, `backend/app/domain/profiles/__init__.py`, profile repositories.
- Vocabulary import/search/enrichment: `backend/app/services/semantic_service.py`, `backend/app/domain/semantics`, semantic graph repositories, Neo4j indexes.
- Task status/progress: `backend/app/core/task_registry.py`.

## Testing

Backend from `backend`:

```powershell
uv run pytest
uv run pytest tests/test_extraction_agents.py
uv run pytest tests/test_semantic_service_vocab_query.py
uv run pytest --cov
```

Frontend from `frontend`:

```powershell
npm.cmd run build
npm.cmd run dev -- --host 127.0.0.1
```
