# SIMONE Frontend

A minimal React/Vite workflow UI for the SIMONE metadata extraction pipeline.

## Run with Docker Compose

From the repository root:

```bash
docker compose --env-file .env.production up -d --build frontend
```

The app is served at:

```text
http://127.0.0.1:3000/
```

The production compose file also starts the full stack when run without a service name:

```bash
docker compose --env-file .env.production up -d --build
```

For development mode, the `docker-compose.dev.yml` file now includes the frontend service while the API still runs locally through the existing startup script.

## Run locally without Docker

```bash
cd frontend
npm install
npm run dev
```

By default the frontend calls `/api/v1`. The Docker image builds with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

For local development with the backend on another origin, create a local environment file with the same variable.

## Current scope

The first vertical slice focuses on the main workflow:

1. Upload a dataset ZIP and inspect package files.
2. Create or refresh chunks.
3. Extract the initial context.
4. Create and validate a DCAT profile draft.
5. Start the existing patching endpoint as a preview of the next workflow stage.

Profiles, vocabularies, and system status are intentionally kept out of the home page for now and should become secondary tabs in the next frontend phase.
