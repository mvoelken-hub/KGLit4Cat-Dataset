---
name: simone
description: Work with the SIMONE thesis prototype and thesis. Use when the user asks to start, verify, inspect, or manage the SIMONE stack; work on SIMONE backend/frontend/codebase tasks; inspect extraction artifacts, models, vocabularies, profiles, or runtime state; write or revise SIMONE thesis material; evaluate prototype/thesis alignment; design experiments or evaluation tables; use NotebookLM-grounded thesis synthesis; manage thesis citations, bibliography, literature status, or LaTeX/PDF build checks.
---

# SIMONE

Use this skill for the local SIMONE repository:

`C:\Users\simcl\Documents\GitHub\Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)`

SIMONE is a thesis prototype for LLM-assisted semantic metadata extraction from uploaded dataset archives. It combines a React/Vite frontend, a FastAPI backend, Neo4j-backed vocabulary search, Ollama-backed embedding/chat models, and a LaTeX thesis under `docs/thesis`.

## Load The Right Reference

Keep this file as the router. Load only the relevant reference:

- Runtime, ports, stack commands, and artifact locations: `references/codebase-map.md`
- Backend/frontend architecture or implementation lookup: `references/codebase-map.md`
- Extraction workflow order, implementation boundaries, or thesis-method accuracy: `references/simone-workflow.md`
- Thesis objective, research questions, scope, contribution framing, or chapter status: `references/thesis-context.md`
- Evaluation dataset, annotations, metrics, baselines, or remaining completeness gaps: `references/evaluation-plan.md`
- NotebookLM, citation traceability, bibliography sync, literature status, or source-grounded prose: `references/notebooklm-workflow.md`

For implementation facts, trust the repo/code/docs over memory. Re-check `docs/WORKFLOW.md`, `docs/thesis/assets/agent-generated-assets/thesis_simone_working_document.md`, and source files when accuracy matters.

## Thesis Source Rule

For substantial thesis prose, literature claims, citation support, chapter rewriting, or source-grounded interpretation:

1. Read `references/notebooklm-workflow.md`.
2. Validate NotebookLM auth with real token fetch.
3. Choose the focused thesis notebook for the topic.
4. Ask NotebookLM for source-grounded context.
5. Map NotebookLM source references to `docs/thesis/bibliography/references.bib`.

Local thesis files and BibTeX entries are useful for structure and style, but they are not a substitute for the NotebookLM pass unless the user explicitly authorizes a no-NotebookLM fallback for that turn.

Read-only implementation/prose alignment audits may rely on repo files, thesis files, and source code without NotebookLM. Switch to NotebookLM when the task judges literature-backed claims, adds/revises thesis prose, or needs citation support.

If NotebookLM auth fails, try `.\scripts\notebooklm-local.ps1 login` yourself once with `SIMONE_NOTEBOOKLM_ALLOW_MUTATION=1`, then rerun the real token-fetch auth check before asking the user to intervene.

Never print cookies, raw auth files, storage state, API keys, or Maton credentials.

## Preferred CLI Workflow

Run SIMONE commands from the repository root. Prefer the repo wrappers over hand-starting services.

Windows:

```powershell
.\simone.bat --help
.\simone.bat status
.\simone.bat dev
.\simone.bat up
.\simone.bat down
```

macOS/Linux:

```bash
./simone --help
./simone status
./simone dev
./simone up
./simone down
```

Use development mode for local code/thesis-adjacent work:

```powershell
.\simone.bat dev
```

Use production mode for Docker API/frontend:

```powershell
.\simone.bat up
.\simone.bat up --build
```

Use infrastructure-only mode for Neo4j/Ollama:

```powershell
.\simone.bat host
.\simone.bat host --neo4j
.\simone.bat host --ollama
```

Stop local SIMONE services with:

```powershell
.\simone.bat down
```

If a service is already reachable, leave it running unless the user asks for restart.

## Common Checks

After startup, verify:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:3000/" -UseBasicParsing -TimeoutSec 5
Invoke-WebRequest -Uri "http://127.0.0.1:8000/docs" -UseBasicParsing -TimeoutSec 5
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 10
```

Report:

- Frontend: `http://127.0.0.1:3000/`
- Backend docs: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/api/v1/health`
- Neo4j Browser: `http://127.0.0.1:7474/browser/` when Neo4j is local

## Thesis And Literature Helpers

Skill-local scripts:

```powershell
.\scripts\notebooklm-local.ps1 auth check --test --json
.\scripts\notebooklm-local.ps1 list --json
python .\scripts\audit-notebook-bib-sync.py --json
python .\scripts\audit-literature-status.py --json
python .\scripts\serpapi-scholar.py "FAIR catalysis metadata ontology" --num 5
.\scripts\thesis-build-check.ps1
```

Run the build helper only when refreshing LaTeX build artifacts under `docs/thesis` is acceptable.

## Testing And Verification

Backend commands from `backend`:

```powershell
uv run pytest
uv run pytest tests/test_extraction_agents.py
uv run pytest tests/test_semantic_service_vocab_query.py
uv run pytest --cov
```

Frontend commands from `frontend`:

```powershell
npm.cmd run build
npm.cmd run dev -- --host 127.0.0.1
```

Use focused tests that match the edited layer. For frontend/API contract changes, run `npm.cmd run build`; for service/domain changes, prefer backend pytest.
