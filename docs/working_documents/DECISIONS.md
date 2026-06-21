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

Reason: Noisy quantitative metadata is harder to remove after projection than missing-but-grounded evidence is to curate or add later. This also keeps the final generated profile more reviewable and reduces low-level parameter clutter. Qualitative labels, placeholder/default values, and encoded enum-like values should not become quantitative attributes merely because they contain numbers.

Tradeoff: Some real extraction targets, such as point counts, resolution values, acquisition dates, or scale factors, may be filtered out of the final draft until the selection rules improve.

Revisit trigger: Revisit when evaluation shows important numeric targets are consistently absent from final drafts, or when curation workload shifts from noise removal to missing-value recovery.

### 2026-06-19: Keep Projection Filters Domain-Agnostic

Decision: Deterministic evidence selection and patching must not use dataset-specific field names, vendor names, instrument names, file names, profile examples, or benchmark sample identifiers as filters.

Reason: Sample-specific filters make evaluation circular and thesis claims indefensible. Projection may use domain-agnostic evidence categories, schema shape, source structure, and broad semantic distinctions such as setting/configuration versus primary data summary.

Tradeoff: Some misclassified evidence may pass through until the category prompt, semantic evaluator, or later reconstruction flow improves.

Revisit trigger: Revisit only to add profile-declared rules or benchmark-independent structural rules, not ad hoc keys from failed examples.

### 2026-06-19: Keep Raw Units Before Vocabulary Normalization

Decision: Profile projection may keep a raw unit string when the source explicitly provides one, but should not pretend local regex cleanup is semantic unit normalization.

Reason: Raw units keep the draft schema-conformant and provide query text for later QUDT/vocabulary normalization. True unit normalization belongs in the grounding stage.

Tradeoff: Raw unit strings may be inconsistent until normalization runs.

Revisit trigger: Revisit when profile schemas support unresolved unit objects or when grounding can reliably update projected unit fields before persistence.

### 2026-06-19: Replace Data Quality Evidence Category With Typed Signals

Decision: Evidence categories are limited to `resource_signal`, `method_signal`, `measurement_signal`, `measurement_condition`, `agent_signal`, `activity_signal`, `instrument_signal`, `surrounding_signal`, and `other`; `data_quality_signal`, `entity_signal`, and category-level `uncertainty` are removed.

Reason: `data_quality_signal` mixed raw data values, metadata, and quality-like notes. The split keeps primary/raw data as evidence-only `measurement_signal`, routes measurement descriptors such as axis bounds, units, ranges, and point counts through `measurement_condition`, routes activities and methods separately, and lets instrument settings become attributes without using raw measurement rows as metadata.

Tradeoff: Some notes that previously influenced profile projection now remain only in evidence artifacts unless they are classified as activity, instrument, resource, method, agent, or surrounding metadata.

Revisit trigger: Revisit when evaluation shows useful final metadata is consistently stranded as inert `measurement_signal` evidence or when `measurement_condition` admits too many row-like observations.

### 2026-06-19: Split Requirement Reporting Into Coverage, Semantics, And Trace

Decision: `requirement_report.json` no longer exposes one `metadata_completeness_score`. It reports deterministic filled-field coverage counts, LLM-assessed `semantic_requirements_score`, and deterministic `source_trace_score`.

Reason: One scalar mixed field presence, semantic adequacy, and source traceability. Filled-field counts are informative without pretending coverage is a quality percentage; semantic quality and trace quality remain separate audit concerns.

Tradeoff: Existing runtime requirement reports are not backward-compatible with the new shape.

Revisit trigger: Revisit when the UI or thesis evaluation needs a deliberately named composite score instead of the three separate metrics.

### 2026-06-19: Separate Coverage Patching From LLM Semantic Reconstruction

Decision: Coverage patching remains evidence-selected and slot-focused; semantic defects are handled afterward by an LLM reconstruction pass that returns constrained JSON Patch operations over the current draft and existing semantic requirement artifacts.

Reason: Semantic issues such as misplaced agents, bloated descriptions, or missing provenance should reorganize the draft rather than trigger another evidence-search patch loop.

Tradeoff: Reconstruction quality depends on the chat model, but deterministic code still enforces allowed paths, schema validation, rollback, and trace records.

Revisit trigger: Revisit when profile-declared reconstruction rules or grounding-stage normalization can replace part of the LLM edit workload.

### 2026-06-20: Observe Description Mining Before Pipeline Integration

Decision: Keep description mining outside the extraction pipeline as a manual, observation-only two-pass probe. Pass one extracts atomic facts from non-distribution description texts only. Deterministic schema search retrieves candidate branches for each fact, and pass two returns concrete JSON Pointer target/value proposals from those candidates without applying them.

Reason: Description fields may contain structured facts that belong in dedicated schema fields, but real model proposals must be inspected before defining deduplication, acceptance, validation, or mutation rules. Description values remain unchanged.

Tradeoff: The probe produces inspectable artifacts but does not improve generated drafts. Routing quality now depends strongly on schema-branch retrieval, and proposed values may still violate candidate object shapes.

Revisit trigger: Integrate it after evidence patching and before semantic evaluation only after observed proposals support clear per-proposal acceptance and validation rules.

### 2026-06-20: Integrate Dataset-Description Facts Through Evidence Patching

Decision: Supersede the observation-only probe. Mine atomic facts only from top-level dataset descriptions after initial draft creation, validate each fact against its exact source text, and add valid facts to a local portable evidence context used by coverage patching. Remove the independent schema-routing pass. Keep semantic evaluation and deterministic source-trace scoring on original source evidence.

Reason: The first mining pass produced compact useful facts, while independent schema routing produced weak targets and invalid value shapes. Existing evidence selection, quantitative grouping, schema validation, collision handling, and rollback already provide the required controlled write path.

Tradeoff: Description-derived facts can fill profile fields but are generated secondary evidence, not direct source-file evidence. They are marked with `draft-description:` provenance and retained in `description_facts.json`, while source-trace scoring excludes them.

Revisit trigger: Revisit provenance linking if generated descriptions gain reliable links to the original evidence IDs that supported them, or when semantic reconstruction is redesigned.

### 2026-06-20: Constrain Patch LLMs With Sliced Target Schemas

Decision: Evidence enrichment, requirement patching, and semantic reconstruction now ask patch LLMs for schema-constrained write envelopes rather than free-form instances or JSON Patch operations. The backend constructs a small output JSON Schema from the allowed target paths and only the reachable `$defs`.

Reason: Profile validation was catching useful but structurally invalid patches too late, causing one invalid operation to consume or roll back useful writes. Target-specific schema slices make invalid value shapes fail at structured-output generation time without sending the full DCAT-AP+ schema.

Tradeoff: The first route supports upsert writes only. Removals, moves, and arbitrary nested JSON Patch edits are intentionally deferred until the schema-constrained route is reliable.

Revisit trigger: Revisit if semantic reconstruction needs validated remove/move operations or if schema slices become too large for specific profile targets.

### 2026-06-20: Simplify Write Envelopes And Separate Attribute Parent Semantics

Decision: Schema-constrained write envelopes use `{writes, reason}`; empty `writes` means no-op. Append writes use canonical array paths instead of `/-`. Semantic requirements now include separate attribute parent semantics, aboutness is fulfilled by either a concrete entity or activity, and aboutness reconstruction uses lean `id`, `title`, and `description` objects.

Reason: `should_apply` duplicated the meaning of empty writes, recursive aboutness schemas caused structured-output failures, and attribute presence needed to be separated from correct parent placement.

Tradeoff: Range splitting remains prompt-led, so a schema-valid collapsed range can still pass if the model emits one.

Revisit trigger: Revisit if evaluation shows prompt-led range handling remains unreliable or if richer profile-declared parent rules replace the current generic routing cues.
