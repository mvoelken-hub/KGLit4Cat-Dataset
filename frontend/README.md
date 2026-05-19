# SIMONE Frontend

A minimal React/Vite workflow UI for the SIMONE metadata extraction pipeline.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

By default the frontend calls `/api/v1`. For local development with the backend on another origin, create a local environment file:

```bash
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## Current scope

The first vertical slice focuses on the main workflow:

1. Upload a dataset ZIP and inspect package files.
2. Create or refresh chunks.
3. Extract the initial context.
4. Create and validate a DCAT profile draft.
5. Start the existing patching endpoint as a preview of the next workflow stage.

Profiles, vocabularies, and system status are intentionally kept out of the home page for now and should become secondary tabs in the next frontend phase.
