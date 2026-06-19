# SIMONE Decisions Ledger

> Snapshot note: This working document records implementation and thesis-facing decisions, with reasons and known tradeoffs. Update it when a decision changes or when a discussion settles a design direction.

## How To Use This File

Record decisions that affect workflow meaning, prototype behavior, evaluation interpretation, or thesis wording. Keep entries short enough to review before writing or changing code.

Each decision should include:

- Date
- Decision
- Reason
- Tradeoff
- Revisit trigger

## Decisions

### 2026-06-19: Default Prototype LLM Calls To Deterministic Generation

Decision: Prototype LLM calls use `OLLAMA_GENERATION_TEMPERATURE=0.0` by default and can enforce a context-derived `num_predict` output cap.

Reason: Evaluation runs should not drift because sampling changed the initial draft. Runaway/repetitive output is bounded by output-token limits instead of nonzero temperature.

Tradeoff: Deterministic decoding can reduce variety in borderline extraction cases.

Revisit trigger: Revisit if a model repeatedly fails structured JSON at temperature 0 despite output caps and repair retries.

### 2026-06-19: Prefer Denoising Over Over-Retention For Numeric Evidence

Decision: The quantitative evidence projection should prefer denoising when uncertain, even if this means some valid numeric facts remain only in evidence/context artifacts instead of becoming final profile attributes.

Reason: Noisy quantitative metadata is harder to remove after projection than missing-but-grounded evidence is to curate or add later. This also keeps the final generated profile more reviewable and reduces low-level parameter clutter.

Tradeoff: Some real extraction targets, such as point counts, resolution values, acquisition dates, or scale factors, may be filtered out of the final draft until the selection rules improve.

Revisit trigger: Revisit when evaluation shows important numeric targets are consistently absent from final drafts, or when curation workload shifts from noise removal to missing-value recovery.

### 2026-06-19: Keep Raw Units Before Vocabulary Normalization

Decision: Profile projection may keep a raw unit string when the source explicitly provides one, but should not pretend local regex cleanup is semantic unit normalization.

Reason: Raw units keep the draft schema-conformant and provide query text for later QUDT/vocabulary normalization. True unit normalization belongs in the grounding stage.

Tradeoff: Raw unit strings may be inconsistent until normalization runs.

Revisit trigger: Revisit when profile schemas support unresolved unit objects or when grounding can reliably update projected unit fields before persistence.

### 2026-06-19: Replace Data Quality Evidence Category With Typed Signals

Decision: Evidence categories are limited to `resource_signal`, `method_signal`, `measurement_signal`, `agent_signal`, `activity_signal`, `instrument_signal`, `surrounding_signal`, and `other`; `data_quality_signal`, `entity_signal`, and category-level `uncertainty` are removed.

Reason: `data_quality_signal` mixed raw data values, metadata, and quality-like notes. The new split keeps primary/raw data as evidence-only `measurement_signal`, routes activities and methods separately, and lets instrument settings become attributes without using raw measurement rows as metadata.

Tradeoff: Some notes that previously influenced profile projection now remain only in evidence artifacts unless they are classified as activity, instrument, resource, method, agent, or surrounding metadata.

Revisit trigger: Revisit when evaluation shows useful final metadata is consistently stranded as inert `measurement_signal` evidence.

### 2026-06-19: Split Requirement Reporting Into Coverage, Semantics, And Trace

Decision: `requirement_report.json` no longer exposes one `metadata_completeness_score`. It reports deterministic `coverage_score`, LLM-assessed `semantic_requirements_score`, and deterministic `source_trace_score`.

Reason: One scalar mixed field presence, semantic adequacy, and source traceability. The split keeps patching focused on coverage while semantic quality and trace quality remain separate audit concerns.

Tradeoff: Existing runtime requirement reports are not backward-compatible with the new shape.

Revisit trigger: Revisit when the UI or thesis evaluation needs a deliberately named composite score instead of the three separate metrics.
