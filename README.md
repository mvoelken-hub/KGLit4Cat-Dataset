# Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)
My thesis work contributes an automated extraction approach that turns heterogeneous catalysis research data into structured, reusable, and FAIR-compliant datasets.

## Research Context

Catalysis research generates heterogeneous experimental data, supporting documents, and domain-specific terminology that are difficult to translate into structured, reusable metadata by hand. This creates friction for documentation, dataset reuse, and later semantic integration with broader research-data infrastructures.

Within the thesis context, this prototype explores whether large language models can support that translation process when combined with explicit workflow artifacts and vocabulary-based grounding. The goal is not to replace expert judgment, but to make metadata construction more traceable, reviewable, and semantically richer than ad-hoc manual extraction alone.

## Prototype Objective

The prototype aims to turn an uploaded dataset archive into a progressively refined metadata representation. It does this by extracting document content, deriving artifact context from the source material, generating an initial metadata draft, refining that draft iteratively, and enriching selected fields with vocabulary-backed semantic references.

In short, the repository serves as an experimental implementation of an LLM-supported semantic metadata extraction pipeline for catalytic experiment resources.

## Running the Full Stack App

Use the helper scripts in the repository root to start the app:

- `up-cpu.bat` starts the CPU setup.
- `up-gpu.bat` starts the GPU setup.

Both scripts let you choose between a production-style Docker setup and a development setup where Neo4j and Ollama run in Docker while the API runs locally.

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
