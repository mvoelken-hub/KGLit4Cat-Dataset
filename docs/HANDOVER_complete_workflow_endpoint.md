# Handover: `codex/complete-workflow-endpoint`

## Branch Purpose

This branch is intended to finish the SIMONE prototype to the point where the implemented application supports the claims made in the thesis, README, and workflow documentation. The target is not general product polish. The target is a coherent thesis prototype: the staged workflow should run end to end, preserve inspectable evidence, support vocabulary-backed semantic grounding, project into a registered metadata profile, and produce enough evaluation artifacts to make the thesis claims defensible.

After this branch is ready, the remaining work should mostly be last-mile usability and operator polish: clearer UI affordances, easier setup, better defaults, and smoother experiment operation. The theoretical workflow, implementation boundaries, and evidence-producing evaluation path should already be in place.

## Current Branch State

The branch already contains the main prototype-completion work:

- A complete workflow endpoint: `POST /api/v1/extraction/workflows/complete`.
- Evaluation CLI support under `simone evaluate run` and `simone evaluate score`.
- Partial evaluation reporting for missing, timed-out, or crashed runs.
- Manual-reference baseline summaries for focused fact-level references.
- Persistence of extraction results, normalization artifacts, warnings, progress, token usage, and run state.
- A merged VPS stack cleanup from PR #4, so the application can run remotely with the frontend/API path working through nginx.

The latest local code direction also removes the LLM file-ranking call and makes file ranking deterministic by default. This is intentional. File ranking should be stable, cheap, explainable, and reproducible for thesis evaluation. The LLM budget should be spent on extraction, normalization, and profile projection rather than on ordering files.

## What The Prototype Must Demonstrate

The thesis and documentation describe SIMONE as a staged assistance system for semantic metadata extraction from heterogeneous scientific data packages. The implementation should therefore demonstrate these claims at prototype level:

- Upload and unpack ZIP-based dataset packages.
- Extract text from supported files, including text files, tables/spreadsheets, PDFs, CSV-like files, and nested archives.
- Retain non-text or weakly interpreted files as resources instead of silently discarding them.
- Chunk extracted text into manageable evidence units.
- Rank/order files and chunks in a deterministic, inspectable way.
- Extract a generic `ExtractionContext` with source traces.
- Normalize quantitative attributes against QUDT quantity kind and unit vocabularies.
- Normalize qualitative attributes against configured vocabularies such as Voc4Cat, CHMO, and nmrCV.
- Persist vocabulary query records and candidate-selection outcomes so semantic grounding can be inspected and scored.
- Project the merged extraction context into a registered metadata profile.
- Validate the final profile document against the profile schema.
- Expose progress, warnings, intermediate state, final result, token usage, and evaluation reports.

The system does not need to prove production autonomy. It may still depend on a reachable Ollama service, Neo4j, imported vocabularies, model authorization, a registered profile, and expert interpretation of evaluation results.

## Highest-Priority Remaining Work

1. Stabilize end-to-end evaluation runs.
   - Use `simone evaluate run --force-rerun` on the current reference datasets.
   - Prioritize completing more than the current successful `IR-IR.zip` run.
   - Treat timeouts and crashes as first-class evidence, but reduce avoidable timeouts where possible.

2. Improve runtime bounds for verbose datasets.
   - The NMR-style package previously timed out around `78/107` chunks.
   - Consider stricter deterministic file/chunk selection, chunk caps for evaluation mode, or resumable batch execution before adding more LLM prompts.
   - Keep any runtime shortcut explicit and documented so the thesis does not overclaim.

3. Strengthen semantic grounding evidence.
   - The preliminary IR report reached schema-valid output but had weak attribute and vocabulary mapping scores.
   - Inspect stored vocabulary query records and normalization artifacts before changing prompts.
   - Prefer fixes that improve traceability and scoring explainability over opaque prompt expansion.

4. Align documentation with implemented behavior.
   - `docs/WORKFLOW.md`, `docs/PROTOTYPE_STATUS.md`, and `docs/evaluation/preliminary_evaluation.md` should describe what the code now actually does.
   - Remove legacy wording about manual patch review or iterative draft refinement where it no longer matches the backend.
   - Keep limitations explicit: schema validity is not semantic correctness, images are not OCR-processed, and the evaluation set is focused rather than thesis-grade exhaustive.

5. Make thesis-claim support evidence-driven.
   - Do not upgrade claims about accuracy, FAIR quality, or superiority over manual curation unless evaluation evidence supports them.
   - Stronger claims currently supported are about architecture, traceability, inspectability, profile validation, and the existence of a repeatable evaluation harness.

## Suggested Developer Workflow

Start from this branch:

```powershell
git checkout codex/complete-workflow-endpoint
git pull --rebase origin codex/complete-workflow-endpoint
```

Run local checks after code changes:

```powershell
cd backend
uv run pytest tests/test_initial_context_extraction.py
uv run pytest tests/test_evaluation.py
uv run pytest tests/test_semantic_service_vocab_query.py
uv run python -m compileall app
```

Run frontend build after API contract or frontend changes:

```powershell
cd frontend
npm.cmd run build
```

Run the stack for long experiments on the VPS when needed:

```bash
ssh root@hostinger-vps
cd /root/SIMONE
git pull
./simone up --build --no-gpu
```

The VPS currently does not expose an NVIDIA Docker runtime, so default GPU startup fails there. Use `--no-gpu` unless the host GPU runtime is fixed. The services are available over Tailscale; this is expected and useful for inspecting the API, Neo4j Browser, Ollama, and frontend during long runs.

## Evaluation Notes

Current evidence is preliminary and should be treated carefully:

- `IR-IR.zip` completed through the automatic endpoint and produced a schema-valid profile document.
- The same IR run had weak semantic quality scores in the preliminary scorer, including no expected qualitative vocabulary mapping hit.
- `1H_NMR-1H_NMR.zip` timed out in a previous one-hour run with partial progress around `78/107` chunks.
- Several reference packages still have missing-output or partial reports rather than completed quality scores.
- The manual-reference baseline is a focused fact-level target, not a complete expert-curated metadata baseline.

The next developer should use these results to guide implementation. Passing schema validation is necessary but not sufficient for the thesis. The important question is whether the staged architecture can recover selected, traceable, semantically grounded facts from representative packages.

## Decision Principles

- Prefer deterministic, inspectable, and reproducible behavior over clever but unstable LLM calls.
- Keep every thesis-relevant workflow stage observable through persisted artifacts or API state.
- Make limitations visible in docs and evaluation outputs instead of hiding them.
- Do not broaden the prototype into a production product before the thesis workflow and evidence are solid.
- When changing prompts, schemas, or evaluation scoring, update tests and documentation in the same commit.

## Ready-To-Merge Criteria For This Branch

Before merging this branch, a new developer should be able to show:

- The complete workflow endpoint can run from upload to final result for at least the core reference package set or explicitly records partial failures.
- Evaluation reports exist for completed and incomplete packages.
- The thesis documentation accurately matches the implemented workflow.
- Deterministic file ranking and chunk ordering are covered by tests.
- Profile projection remains schema-valid for successful runs.
- Vocabulary normalization artifacts are persisted and scoreable.
- No major thesis claim depends only on aspiration rather than code, docs, or evaluation evidence.

