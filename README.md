# Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)

My thesis work contributes an automated extraction approach that turns heterogeneous catalysis research data into structured, reusable, and FAIR-compliant datasets.

## Research Context

Catalysis research generates heterogeneous experimental data, supporting documents, and domain-specific terminology that are difficult to translate into structured, reusable metadata by hand. This creates friction for documentation, dataset reuse, and later semantic integration with broader research-data infrastructures.

Within the thesis context, this prototype explores whether large language models can support that translation process when combined with explicit workflow artifacts and vocabulary-based grounding. The goal is not to replace expert judgment, but to make metadata construction more traceable, reviewable, and semantically richer than ad-hoc manual extraction alone.

## Prototype Objective

The prototype aims to turn an uploaded dataset archive into a progressively refined metadata representation. It does this by extracting document content, deriving artifact context from the source material, generating an initial metadata draft, refining that draft iteratively, and enriching selected fields with vocabulary-backed semantic references.

In short, the repository serves as an experimental implementation of an LLM-supported semantic metadata extraction pipeline for catalytic experiment resources.

## Running the App

### Prerequisites

| Tool | Required for | Notes |
|---|---|---|
| Docker Desktop | All modes | Provides Docker Compose and runs Neo4j, Ollama, API, and frontend containers. |
| uv | All modes | Used by the `simone` CLI wrapper and the local development API. |
| Node.js + npm | Optional | Only needed if you want to run the frontend locally with hot reload. Without npm, `simone dev` falls back to the Docker frontend. |

Install links:

- Docker Desktop: https://www.docker.com/get-started/
- uv: https://docs.astral.sh/uv/getting-started/installation/
- Node.js: https://nodejs.org/

### Clone and Prepare

```bash
git clone https://github.com/smnclmns/Semantic-Inference-Module-for-Ontology-driven-Node-Extraction-SIMONE-.git
cd Semantic-Inference-Module-for-Ontology-driven-Node-Extraction-SIMONE-
```

The CLI creates `.env.production` or `.env.development` from the matching example file on first use.

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

Production mode runs the full stack in Docker: Neo4j, Ollama, API, and frontend.

```bash
simone up
```

Use GPU support for Ollama:

```bash
simone up --gpu
```

By default, `up` reuses existing images and containers. Rebuild images explicitly after Dockerfile or dependency changes:

```bash
simone up --build
```

### Development Mode

Development mode runs the API locally with Uvicorn reload. Neo4j and Ollama run in Docker.

```bash
simone dev
```

If npm is installed, the frontend runs locally with Vite hot reload. If npm is missing, the CLI prints the Node.js download link and falls back to Docker frontend mode automatically.

You can choose Docker frontend mode directly:

```bash
simone dev --no-npm
```

Use GPU support for Ollama in dev mode:

```bash
simone dev --gpu
```

### Service URLs

| Service | URL |
|---|---|
| Frontend | http://127.0.0.1:3000 |
| API Docs | http://127.0.0.1:8000/docs |
| API Health | http://127.0.0.1:8000/api/v1/health |
| Neo4j Browser | http://127.0.0.1:7474/browser/ |

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

The reset command asks whether to create a backup before deleting the data. Non-interactive variants are available:

```bash
simone reset-neo4j --backup
simone reset-neo4j --yes
```

Restore a backup:

```bash
simone restore-neo4j .backups/neo4j/<backup-file>.cypher
```

