# Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)

My thesis work contributes an automated extraction approach that turns heterogeneous catalysis research data into structured, reusable, and FAIR-compliant datasets.

## Research Context

Catalysis research generates heterogeneous experimental data, supporting documents, and domain-specific terminology that are difficult to translate into structured, reusable metadata by hand. This creates friction for documentation, dataset reuse, and later semantic integration with broader research-data infrastructures.

Within the thesis context, this prototype explores whether large language models can support that translation process when combined with explicit workflow artifacts and vocabulary-based grounding. The goal is not to replace expert judgment, but to make metadata construction more traceable, reviewable, and semantically richer than ad-hoc manual extraction alone.

## Prototype Objective

The prototype aims to turn an uploaded dataset archive into a progressively refined metadata representation. It does this by extracting document content, deriving artifact context from the source material, generating an initial metadata draft, refining that draft iteratively, and enriching selected fields with vocabulary-backed semantic references.

The active backend workflow is direct extraction, vocabulary normalization, profile projection, validation, and persistence. Some older "initial draft" and "patch review" names still exist in the frontend for compatibility, but manual patch review is no longer the active backend pipeline.

## Running the App

### Prerequisites

| Tool | Required for | Notes |
|---|---|---|
| [Docker Desktop](https://www.docker.com/get-started/) | All modes | Provides Docker Compose and runs Neo4j, Ollama, API, and frontend containers. |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | All modes | Used by the `simone` CLI wrapper and the local development API. |
| [Node.js + npm](https://nodejs.org/) | Optional | Only needed if you want to run the frontend locally with hot reload. Without npm, `simone dev` falls back to the Docker frontend. |

### Clone and Prepare

```bash
git clone https://github.com/smnclmns/Semantic-Inference-Module-for-Ontology-driven-Node-Extraction-SIMONE.git
cd Semantic-Inference-Module-for-Ontology-driven-Node-Extraction-SIMONE
```


### Environment Variables

The CLI creates one `.env` file from `.env.example` and sets `APP_ENV` automatically: `simone up` runs in production mode and `simone dev` runs in development mode.

| Variable | Production default | Development default | Purpose |
|---|---|---|---|
| `APP_ENV` | `production` | `development` | Runtime mode. The CLI sets this automatically for `simone up` and `simone dev`. |
| `NEO4J_HOSTNAME` | empty | empty | Optional Neo4j host override. Empty, `localhost`, or `127.0.0.1` means the CLI manages the local Neo4j Docker service. Remote values are used directly and the local service is skipped. |
| `NEO4J_PORT` | `7687` | `7687` | Neo4j Bolt port. The API derives the full Bolt URI automatically. |
| `NEO4J_USER` | `neo4j` | `neo4j` | Neo4j username.\* |
| `NEO4J_PASSWORD` | `12345678` | `12345678` | Neo4j password used by the API and container initialization.\* |
| `OLLAMA_HOSTNAME` | empty | empty | Optional Ollama host override. Empty, `localhost`, or `127.0.0.1` means the CLI manages the local Ollama Docker service. Remote values are used directly and the local service is skipped. |
| `OLLAMA_PORT` | `11433` | `11433` | Ollama API port. The API derives the base URL automatically. |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | `qwen3-embedding:0.6b` | Embedding model used for semantic search and vocabulary grounding. |
| `OLLAMA_CHAT_MODEL` | `gemma4:31b-cloud` | `gemma4:31b-cloud` | Chat model used by extraction agents. The default is an Ollama Cloud model and may require sign-in. |
| `OLLAMA_EMBED_DIMENSIONS` | `768` | `768` | Expected embedding vector size for the configured embedding model. |
| `EMBEDDING_BATCH_SIZE` | `32` | `32` | Number of texts embedded per Ollama request batch. |
| `MAX_CONTEXT_LENGTH` | `64000` | `64000` | Maximum model context length used when preparing Ollama requests. |
| `RUNTIME_DIR` | `./.runtime` | `./.runtime` | Base runtime directory for uploads, vocabularies, profiles, output files, and API logs. |
| `FRONTEND_PORT` | `3000` | `3000` | Frontend browser port. The API automatically allows `localhost` and `127.0.0.1` origins for this port. |
| `SKIP_MODEL_PULL` | `false` | `true` | Override: skip pulling configured Ollama models on startup. |
| `SKIP_INITIAL_VOCAB_IMPORT` | `false` | `true` | Override: skip importing initial vocabularies on startup. |

\*Neo4j applies `NEO4J_USER` and `NEO4J_PASSWORD` only when `data/docker/neo4j/data` is initialized for the first time. If you change either value later, either update the `.env` file to match the persisted database credentials or run `simone reset-neo4j`.

Startup behavior is mode-specific by default. Production **pulls configured models**, **imports initial vocabularies**, and **generates missing embeddings**.

Development skips those startup-heavy tasks. Advanced users can override this by adding `SKIP_INITIAL_VOCAB_IMPORT` or `SKIP_MODEL_PULL` to `.env`.

The initial vocabulary list is defined in `backend/app/core/initial_vocabs.py`. Adjust `INITIAL_VOCABS` there if you want SIMONE to bootstrap a different set of vocabularies. You can inspect the active configuration with:

```bash
simone vocabs --info
```

Configured Ollama models come from `OLLAMA_EMBED_MODEL` and `OLLAMA_CHAT_MODEL` in `.env`. Run model management on the machine that hosts Ollama:

```bash
simone models
```

When `.env` points to a remote Ollama host, the local SIMONE CLI only inspects that host. Pulling, removing, selecting resident models, and server-side memory tuning should be done on the Ollama host. The web UI can adjust runtime request options for future SIMONE agent calls, but it does not write `.env` or change Docker/Ollama host settings.


### CLI Wrappers

Run commands from the repository root:

```bash
# Windows
simone.bat --help

# macOS / Linux
./simone --help
```

If your shell can resolve the wrapper as `simone`, you can use `simone` instead of `simone.bat` or `./simone`.

### Production Mode

Production mode runs the API and frontend in Docker. Neo4j and Ollama are also started as local Docker services unless `.env` points them at remote hosts. Ollama is configured with GPU acceleration by default.

```bash
simone up
```

By default, `up` reuses existing images and containers. Rebuild images explicitly after Dockerfile or dependency changes:

```bash
simone up --build
```

### Development Mode

Development mode runs the API locally with Uvicorn reload. Neo4j and Ollama run in Docker unless `.env` points them at remote hosts.

```bash
simone dev
```

If npm is installed, the frontend runs locally with Vite hot reload. If npm is missing, the CLI prints the Node.js download link and falls back to Docker frontend mode automatically.

You can choose Docker frontend mode directly:

```bash
simone dev --no-npm
```

To attach both local API and frontend logs to the terminal where you ran the command, use:

```bash
simone dev -fg
```

With local npm frontend mode, Vite and Uvicorn both write to the same terminal. To use the Docker frontend and follow its logs instead, combine it with Docker frontend mode:

```bash
simone dev -fg --no-npm
```


### Infrastructure Mode

Infrastructure mode starts only Neo4j and/or Ollama on a host machine, without the API or frontend. This is useful when you want another machine to run `simone dev` or `simone up` while connecting to these services remotely.

On the host machine, leave `NEO4J_HOSTNAME` and `OLLAMA_HOSTNAME` empty in `.env` so the CLI starts the local Docker services:

```bash
# Start all local infrastructure services
simone host

# Start only Neo4j
simone host --neo4j

# Start only Ollama
simone host --ollama
```

The command prints the `.env` values that other machines should use to connect. For example, on a Tailscale network:

```env
NEO4J_HOSTNAME=<my-gpu-box>
NEO4J_PORT=7687
OLLAMA_HOSTNAME=<my-gpu-box>
OLLAMA_PORT=11433
```

If both hostnames are set to remote addresses in `.env`, `simone host` exits with a message — there is nothing to start locally.

### Service URLs

| Service | URL |
|---|---|
| Frontend | http://127.0.0.1:3000 by default, or the configured `FRONTEND_PORT` |
| API Docs | http://127.0.0.1:8000/docs |
| API Health | http://127.0.0.1:8000/api/v1/health |
| Neo4j Browser | http://127.0.0.1:7474/browser/ |

### Complete Workflow Endpoint

After the API is running and a profile is registered, one multipart endpoint can upload a ZIP package and let the backend run chunking plus extraction automatically:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/extraction/workflows/complete" \
  -F "file=@./my-dataset.zip" \
  -F "profile_identifier=dcat-ap-plus"
```

The endpoint returns the deterministic data package id, workflow status, and URLs for polling progress and retrieving the final result:

- `GET /api/v1/extraction/run/{data_package_id}/progress`
- `GET /api/v1/extraction/result/{data_package_id}`

### Stop and Inspect

```bash
simone status
simone down
```

`status` shows Docker containers, local dev processes, and API health. `down` stops SIMONE containers and closes local development API/frontend processes.

### Ollama Cloud Sign-In

The default chat model is an Ollama Cloud model (`gemma4:31b-cloud`). On a fresh machine or after resetting the Ollama data volume, `/api/v1/health` may report the chat check as `unauthorized (status code: 401)`.

`simone up` checks health after the API is reachable and starts the Ollama sign-in flow when needed. You can also run it manually:

```bash
simone signin-ollama
```

If Ollama prints a connect link, open it in your browser and complete the sign-in. The credential is stored in `data/docker/ollama/data`, so you normally only need this once per local data volume.

### Neo4j Data Maintenance

Neo4j stores local data in `data/docker/neo4j/data`. The password is applied only when this data directory is initialized. Changing `NEO4J_PASSWORD` later does not change the persisted database password.

Create a backup:

```bash
simone backup-neo4j
```

Reset local Neo4j data:

```bash
simone reset-neo4j
```


Restore a backup:

```bash
simone restore-neo4j .backups/neo4j/<backup-file>.cypher
```
