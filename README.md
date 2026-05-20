# Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)
My thesis work contributes an automated extraction approach that turns heterogeneous catalysis research data into structured, reusable, and FAIR-compliant datasets.

## Research Context

Catalysis research generates heterogeneous experimental data, supporting documents, and domain-specific terminology that are difficult to translate into structured, reusable metadata by hand. This creates friction for documentation, dataset reuse, and later semantic integration with broader research-data infrastructures.

Within the thesis context, this prototype explores whether large language models can support that translation process when combined with explicit workflow artifacts and vocabulary-based grounding. The goal is not to replace expert judgment, but to make metadata construction more traceable, reviewable, and semantically richer than ad-hoc manual extraction alone.

## Prototype Objective

The prototype aims to turn an uploaded dataset archive into a progressively refined metadata representation. It does this by extracting document content, deriving artifact context from the source material, generating an initial metadata draft, refining that draft iteratively, and enriching selected fields with vocabulary-backed semantic references.

In short, the repository serves as an experimental implementation of an LLM-supported semantic metadata extraction pipeline for catalytic experiment resources.

## Running the Full Stack App

### Quick Start

Clone the repo:
```bash
git clone https://github.com/smnclmns/Semantic-Inference-Module-for-Ontology-driven-Node-Extraction-SIMONE-.git
```

After cloning the repository, run the setup assistant from the repo root:

```bash
python setup.py
```

This script will:
1. Check if Docker Desktop is installed — if not, it gives you the download link.
2. Check if `uv` is installed — if not, it gives you the install command.
3. Check if Node.js is installed (optional, only needed for dev mode).
4. Run `uv sync` to install Python dependencies.
5. Print the commands to start the app.

---

### Manual Prerequisites

If you prefer to install manually:

| Tool | Required for | Download |
|---|---|---|
| **Docker Desktop** | All modes | [docker.com/get-started](https://www.docker.com/get-started/) |
| **uv** | All modes | [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| **Node.js + npm** | `simone dev` only | [nodejs.org](https://nodejs.org/) |

---

### Start the App

**Production mode** (all services in Docker — no Node.js needed):

```bash
# Windows
simone.bat up

# macOS / Linux
./simone up
```

**Development mode** (Neo4j + Ollama in Docker; API + frontend locally with hot reload):

```bash
# Windows
simone.bat dev

# macOS / Linux
./simone dev
```

**With GPU support** (append `--gpu` to either command):

```bash
# Windows
simone.bat up --gpu
simone.bat dev --gpu

# macOS / Linux
./simone up --gpu
./simone dev --gpu
```

---

### Access the Services

Once started, the services are available at:

| Service | URL |
|---|---|
| Frontend | http://127.0.0.1:3000 |
| API Docs | http://127.0.0.1:8000/docs |
| Neo4j Browser | http://127.0.0.1:7474/browser/ |

---

### Stop and Status

```bash
# Windows
simone.bat down     # Stop all services
simone.bat status   # Check what's running

# macOS / Linux
./simone down
./simone status
```

---

### First-Time Ollama Cloud Sign-In

The default chat model is an Ollama Cloud model (`gemma4:31b-cloud`). On a fresh machine or after resetting the Ollama data volume, the app can start successfully but the `/api/v1/health` endpoint may report the chat check as `unauthorized (status code: 401)` until the Ollama container is signed in.

After starting the stack once, run:

```bat
scripts\signin-ollama.bat
```

The script runs `ollama signin` inside the running Ollama Docker container. If the container is not signed in yet, Ollama prints a link like:

```text
https://ollama.com/connect?name=...&key=...
```

Open that link in your browser and complete the sign-in. The credential is stored in the mounted Ollama data directory (`data/docker/ollama/data`), so you normally only need to do this once per local data volume.

If you prefer to run the command manually, use:

```bat
docker compose exec -T ollama ollama signin
```
