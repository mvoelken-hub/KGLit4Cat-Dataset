# SIMONE Preliminary Prototype Evaluation

This note records the current evidence for the prototype claims after adding the complete workflow endpoint and the first evaluation harness. It is intentionally preliminary: the references are small, hand-authored checks for representative packages in `data/datasets`, not a thesis-grade benchmark.

## Implemented Evaluation Support

- `simone evaluate run` submits ZIP datasets to `POST /api/v1/extraction/workflows/complete`, polls until a result is available, stores run manifests, and scores completed outputs.
- `simone evaluate run --force-rerun` clears stale extraction artifacts for deterministic package ids before scheduling a new run, which is needed when evaluating code changes against a package that already completed earlier.
- `simone evaluate score` scores existing runtime outputs under `backend/.runtime/output` against reference annotations.
- Reference annotations live under `data/evaluation/references` and cover expected files, objects, attributes, vocabulary mappings, and required projected profile fields.
- The final extraction result now persists the normalization artifact, so vocabulary grounding can be scored after the workflow completes.
- The extraction context now receives a deterministic resource inventory for every file in the uploaded package, including files that were not semantically interpreted by text extraction.

## Current Real-Run Evidence

| Dataset | Outcome | Evidence |
| --- | --- | --- |
| `IR-IR.zip` | Completed through the automatic endpoint after a post-fix `--force-rerun`. | The generated report was schema-valid with top-1 file ranking hit, top-3 file recall `0.75`, source trace coverage `1.0`, profile field coverage `1.0`, object macro F1 `0.2667`, attribute F1 `0.0`, and vocabulary mapping F1 `0.0`. |
| `1H_NMR-1H_NMR.zip` | Did not finish within a one-hour evaluation timeout. | The workflow was still in chunk extraction at `78/107` chunks when paused. This is evidence of a runtime/scalability failure mode for small but verbose NMR packages. |

The IR result shows that the complete endpoint can produce a schema-valid profile artifact and enough persisted state for scoring. The same result also shows that schema validity does not imply scientific or semantic quality: the reference method `IR`, expected qualitative mapping, and expected key attributes were not recovered by the current scorer.

## Claim Support Assessment

Supported at prototype level:

- A non-UI complete workflow exists for upload, chunking, extraction, normalization, projection, validation, and result polling.
- The prototype can persist traceable extraction artifacts and score them after completion.
- Schema-valid profile projection is reachable on at least one real package, `IR-IR.zip`.
- File ranking and source trace capture are measurable rather than only described.

Not yet substantiated:

- Extraction quality across representative catalysis and spectroscopy packages.
- Robust vocabulary grounding quality. The first IR report produced no expected qualitative mapping hit.
- Bounded runtime for NMR-style packages with many text-bearing files and repeated parameter blocks.
- General claims about scientific correctness, FAIR metadata quality, or cross-dataset robustness.

## Next Evaluation Work

- Add runtime controls for package/chunk limits or more selective chunk ranking before expensive LLM extraction.
- Expand reference annotations with expert-reviewed expected objects and accepted vocabulary URIs.
- Record final thesis tables only after multiple post-fix completed runs.
