# Spec: Evidence Grounding for Synthesized Semantic Relations

## Problem

The semantic reconstruction pipeline synthesizes new relation objects
(`evaluated_activity`, `is_about_activity`) via LLM calls when the LLM
diagnosis identifies a "missing" relation.  These synthesized objects
are frequently **not grounded in evidence** — they have generic
placeholder ids, invented titles, and no traceable connection to source
text.  This causes:

1. **False `partial` on `dataset_subject_evaluation_distinction`** — the
   requirement checks that `is_about_*` and `evaluated_*` relations are
   "independently supported claims," but the synthesized relations are
   LLM inventions, not evidence-backed claims.  The evaluator correctly
   rates `partial`, but the system cannot fix it because the fix IS the
   problem.

2. **Polluted draft with ungrounded relations** — the final draft
   contains objects like `{"id": "is_about_activity", "title":
   "Infrared spectrum analysis"}` whose id is a literal placeholder and
   whose title is synthesized from the dataset context rather than from
   any evidence span.

3. **No traceability** — a human reader cannot determine which evidence
   span supports a synthesized relation, because none does.

## Root Cause Analysis

### Pipeline flow (current)

```
_evaluate_semantic_requirements()
  → LLM evaluator rates dataset_subject_evaluation_distinction as "partial"
  → _guard_semantic_requirement_assessment() does NOT override
     (guard only checks applicability, not evidence grounding)
  → status remains "partial"

_reconstruct_semantic_defects()
  → _semantic_reconstruction_update() for each non-fulfilled requirement
    → _compile_deterministic_semantic_actions() → no deterministic actions
    → falls through to LLM diagnosis via _compile_semantic_diagnosis_actions()
      → LLM diagnosis produces defects:
          defect_type="bad_aboutness", target="/is_about_activity",
            recommended_action="append", needs_synthesis=true
          defect_type="bad_provenance", target="/was_generated_by/0/evaluated_activity",
            recommended_action="append", needs_synthesis=true
      → For each defect with needs_synthesis=true:
        → _synthesize_semantic_reconstruction_write()
          → LLM generates a new object (title, description, type, id)
          → _semantic_absence_placeholder() checks for "no evidence" phrases
          → If not a placeholder, the object is appended to the draft
```

### Why the safeguards fail

| Safeguard | Location | Why it fails |
|---|---|---|
| `_semantic_absence_placeholder()` | `projection_service.py:2893` | Only checks for explicit "no evidence" / "not available" phrases. An LLM that generates `{"id": "activity:IR_spectrum_analysis", "title": "IR spectrum analysis"}` passes this check because the text doesn't contain absence phrases. |
| `_relation_item_looks_like_requirement_placeholder()` | `projection_service.py:2374` | Only rejects ids that match requirement_id names (e.g., `"aboutness_concreteness"`). Generic synthesized ids like `"is_about_activity"` or `"activity:IR_spectrum_analysis"` pass through. |
| `SEMANTIC_SYNTHESIS_SYSTEM_PROMPT` | `requirement_enrichment.py:704` | Says "Do not invent facts" but the LLM has no structured way to signal "evidence insufficient" other than returning an absence-placeholder phrase, which it often doesn't do. |
| `dataset_subject_evaluation_distinction` guard | `projection_service.py:2873` | Only checks whether the distinction is *applicable* (both relation levels present). It does not check whether the relations are *evidence-grounded*. |

### What the current synthesis prompt provides to the LLM

```python
# _semantic_synthesis_prompt() payload:
{
    "requirement": {...},
    "defect": {
        "defect_type": "bad_aboutness",
        "target_path": "/is_about_activity",
        "recommended_action": "append",
        "needs_synthesis": true,
        "reason": "The dataset has an evaluated_entity but lacks an
                   explicit is_about_activity relation..."
    },
    "current_target": null,   # array is empty
    "selected_evidence": [...],   # evidence items
    "context_window": [...],
    "rules": [
        "Return one small schema-valid object or value for the defect target.",
        "Use only selected evidence and context window.",
        "Do not include profile patch operations."
    ]
}
```

The LLM is told to "use only selected evidence" but:
- There is no structured field that says "this evidence span supports
  this relation"
- The LLM cannot return a structured "refuse — evidence insufficient"
  response (it can only return an absence-placeholder text, which is
  fragile)
- There is no backend post-check that the LLM's output fields actually
  match any evidence span

## Target State

After this work:

1. **Synthesized relations must be evidence-grounded or refused.** When
   the LLM synthesizes a relation object (e.g., for `is_about_activity`
   or `evaluated_activity`), the backend must verify that at least one
   field value (title or description) contains text traceable to a
   selected evidence span.  If no evidence match is found, the synthesis
   is rejected and the defect remains unresolved with a clear rationale.

2. **Synthesized relation ids must be namespace-qualified and
   traceable.** Generic placeholder ids (`"is_about_activity"`,
   `"activity:foo"`) must be detected and rejected.  The backend should
   generate a deterministic id from the data package id + evidence
   candidate id when the LLM fails to provide a grounded one.

3. **The `dataset_subject_evaluation_distinction` guard must detect
   ungrounded relations.** After reconstruction, the guard should check
   whether `is_about_activity` and `evaluated_activity` entries are
   evidence-grounded.  If they are not, the status should be `partial`
   with a rationale explaining which relation lacks evidence support.

4. **The synthesis LLM must have a structured "refuse" pathway.** The
   synthesis prompt and output schema must include a way for the LLM to
   return "evidence_insufficient" as a structured response, not just a
   text phrase.

5. **No regression on existing fulfilled requirements.** The 12
   currently-fulfilled requirements must remain fulfilled.  The
   `dataset_subject_evaluation_distinction` requirement should either
   become `fulfilled` (if evidence grounding is confirmed) or remain
   `partial` with an honest, specific rationale.

## Implementation Plan

### Phase 1: Evidence-grounding validation for synthesized relations

**File:** `backend/app/services/projection_service.py`
**Method:** New `_validate_synthesis_evidence_grounding()`

After `_synthesize_semantic_reconstruction_write()` returns a write,
validate that the synthesized object's `title` and/or `description`
contain at least one significant phrase (3+ characters, non-stop-word)
that appears in the selected evidence text.

```python
@classmethod
def _validate_synthesis_evidence_grounding(
    cls,
    *,
    synthesized_value: dict[str, Any],
    selected_evidence: list[RequirementEvidenceItem],
    min_matched_phrases: int = 1,
) -> list[str]:
    """Return rejection reasons if synthesized value has no evidence grounding."""
    # Extract significant phrases from the synthesized object's text fields
    text_fields = []
    for key in ("title", "description"):
        val = synthesized_value.get(key)
        if isinstance(val, list):
            text_fields.extend(str(v) for v in val)
        elif val:
            text_fields.append(str(val))
    if not text_fields:
        return ["Synthesized relation has no text fields to ground in evidence."]
    
    syn_text = cls._normalized_text(" ".join(text_fields))
    syn_phrases = {
        w for w in syn_text.split()
        if len(w) >= 3 and w not in cls._LABEL_CLEAN_STOP_WORDS
    }
    if not syn_phrases:
        return ["Synthesized relation text has no significant phrases."]
    
    # Collect all evidence text
    evidence_phrases: set[str] = set()
    for ev in selected_evidence:
        for field in ("claim", "evidence_text", "source_context"):
            text = cls._normalized_text(getattr(ev, field, "") or "")
            evidence_phrases.update(
                w for w in text.split()
                if len(w) >= 3 and w not in cls._LABEL_CLEAN_STOP_WORDS
            )
    
    if not evidence_phrases:
        return ["No evidence text available for grounding check."]
    
    matched = syn_phrases & evidence_phrases
    if len(matched) < min_matched_phrases:
        return [
            f"Synthesized relation text does not match any evidence: "
            f"synthesized phrases {syn_phrases} have no overlap with evidence."
        ]
    return []
```

**Integration point:** In `_compile_semantic_diagnosis_actions()`, after
line ~1300 where `write, synth_rejected = await
self._synthesize_semantic_reconstruction_write(...)` returns
successfully, add:

```python
if write is not None and write.mode == "append" and write.items:
    for item_in_write in write.items:
        if isinstance(item_in_write, dict) and any(
            k in item_in_write for k in ("title", "description", "type")
        ):
            grounding_rejections = cls._validate_synthesis_evidence_grounding(
                synthesized_value=item_in_write,
                selected_evidence=item.selected_evidence,
            )
            if grounding_rejections:
                rejected.extend(grounding_rejections)
                write = None
                break
```

### Phase 2: Stronger placeholder/id detection

**File:** `backend/app/services/projection_service.py`
**Method:** Extend `_relation_item_looks_like_requirement_placeholder()`

Add checks for:
- Ids that equal the path component name (e.g., `"is_about_activity"`)
- Ids without a namespace separator (`:`) that are lowercase
  descriptions rather than identifiers
- Titles that are generic activity descriptions without evidence
  specificity (e.g., "Activity that generated the evaluated entity")

```python
@staticmethod
def _relation_item_looks_like_requirement_placeholder(item: dict[str, Any]) -> bool:
    # ... existing checks ...
    identifier = str(item.get("id") or "").strip().lower()
    
    # New: reject ids that match relation path component names
    relation_path_names = {
        "is_about_activity", "is_about_entity",
        "evaluated_activity", "evaluated_entity",
        "carried_out_by",
    }
    if identifier in relation_path_names:
        return True
    
    # New: reject lowercase ids without namespace separator
    # that look like descriptions, not identifiers
    if identifier and ":" not in identifier and " " in identifier:
        return True
    
    # New: reject generic description patterns
    description = str(item.get("description") or "").strip().lower()
    generic_patterns = [
        "activity that generated",
        "activity that measured",
        "activity that analysed",
        "no specific activity",
    ]
    if any(p in description for p in generic_patterns):
        return True
    
    return False
```

### Phase 3: Structured "refuse" pathway for synthesis LLM

**File:** `backend/app/domain/extraction/requirement_enrichment.py`

Update `SEMANTIC_SYNTHESIS_SYSTEM_PROMPT` to include:

```
If the supplied evidence does not directly support the target relation,
return {"evidence_insufficient": true, "reason": "<explanation>"}.
Do not fabricate a relation object when evidence is insufficient.
```

**File:** `backend/app/services/projection_service.py`
**Method:** Update `_synthesize_semantic_reconstruction_write()`

After `value = result.output`, check for the `evidence_insufficient`
field:

```python
if isinstance(value, dict) and value.get("evidence_insufficient"):
    return None, [
        f"Synthesis LLM refused: evidence insufficient at {defect.target_path}. "
        f"Reason: {value.get('reason', 'unspecified')}"
    ]
```

Also update `_semantic_absence_placeholder()` to detect
`evidence_insufficient: true` in the synthesized value.

### Phase 4: Post-reconstruction evidence grounding guard

**File:** `backend/app/services/projection_service.py`
**Method:** Extend `_guard_semantic_requirement_assessment()` for
`dataset_subject_evaluation_distinction`

After reconstruction, re-check whether the relations are
evidence-grounded:

```python
if requirement.requirement_id == "dataset_subject_evaluation_distinction":
    if not cls._subject_target_distinction_applicable(document):
        item.status = "not_applicable"
        ...
    else:
        # Check each relation for evidence grounding
        ungrounded = cls._ungrounded_relations(document, item.selected_evidence)
        if ungrounded:
            item.status = "partial"
            item.quality = 0.5
            item.applicable = True
            item.rationale = (
                f"Backend inspection found {len(ungrounded)} relation(s) "
                f"without evidence grounding: {', '.join(ungrounded[:3])}."
            )
        elif item.status in {"missing", "not_applicable", "partial"}:
            # Both relation levels present and grounded
            item.status = "fulfilled"
            item.quality = 1.0
            item.applicable = True
            item.rationale = (
                "Both dataset subject and activity evaluation-target "
                "relations are present and evidence-grounded."
            )
```

**New method:** `_ungrounded_relations()`

```python
@classmethod
def _ungrounded_relations(
    cls,
    document: dict[str, Any],
    evidence: list[RequirementEvidenceItem],
) -> list[str]:
    """Return labels of relation entries that lack evidence grounding."""
    ungrounded: list[str] = []
    relations_to_check = [
        ("is_about_entity", "/is_about_entity"),
        ("is_about_activity", "/is_about_activity"),
    ]
    # Also check evaluated_entity/evaluated_activity on each activity
    for i, activity in enumerate(document.get("was_generated_by") or []):
        if not isinstance(activity, dict):
            continue
        for rel in ("evaluated_entity", "evaluated_activity"):
            for j, entry in enumerate(activity.get(rel) or []):
                if not isinstance(entry, dict):
                    continue
                if cls._relation_item_looks_like_requirement_placeholder(entry):
                    ungrounded.append(f"/was_generated_by/{i}/{rel}/{j}")
                    continue
                grounding = cls._validate_synthesis_evidence_grounding(
                    synthesized_value=entry,
                    selected_evidence=evidence,
                )
                if grounding:
                    ungrounded.append(f"/was_generated_by/{i}/{rel}/{j}")
    # Check is_about_entity and is_about_activity at document level
    for key, path in relations_to_check:
        for j, entry in enumerate(document.get(key) or []):
            if not isinstance(entry, dict):
                continue
            if cls._relation_item_looks_like_requirement_placeholder(entry):
                ungrounded.append(f"{path}/{j}")
                continue
            grounding = cls._validate_synthesis_evidence_grounding(
                synthesized_value=entry,
                selected_evidence=evidence,
            )
            if grounding:
                ungrounded.append(f"{path}/{j}")
    return ungrounded
```

### Phase 5: Cleanup ungrounded synthesized relations

**File:** `backend/app/services/projection_service.py`
**Method:** Add to post-reconstruction cleanup

After the existing cleanup (duplicate coherence, parent placement, label
quality), add a pass that removes ungrounded synthesized relations:

```python
# Post-reconstruction: remove ungrounded synthesized relations
ungrounded = cls._ungrounded_relations(current, ...)
if ungrounded:
    for path in sorted(ungrounded, reverse=True):  # descending for safe removal
        cleanup_writes.append(SchemaConstrainedWrite(
            target_path=path,
            mode="remove",
            reason="Remove ungrounded synthesized relation lacking evidence support.",
        ))
```

**Important:** This cleanup must only remove entries that were
synthesized by the reconstruction pipeline, not entries that were
present in the initial extraction.  To distinguish them, check whether
the entry's id or title appears in the pre-reconstruction document.  If
it does, it was extracted (not synthesized) and should be kept even if
evidence grounding is weak.

### Phase 6: Re-evaluate after reconstruction

**File:** `backend/app/services/projection_service.py`
**Method:** After `_reconstruct_semantic_defects()`, re-run the guard
for `dataset_subject_evaluation_distinction` on the final document.

Currently the guard runs only during initial evaluation (before
reconstruction).  After reconstruction and cleanup, re-run the guard
so the status reflects the post-reconstruction state:

```python
# After all reconstruction and cleanup:
for item in semantic_items:
    if item.requirement_id == "dataset_subject_evaluation_distinction":
        self._guard_semantic_requirement_assessment(
            requirement=requirement,
            document=current,   # post-reconstruction document
            item=item,
        )
```

## Files to Modify

| File | Changes |
|---|---|
| `backend/app/services/projection_service.py` | Phases 1, 2, 4, 5, 6 — new validation methods, extended placeholder detection, post-reconstruction guard, cleanup pass |
| `backend/app/domain/extraction/requirement_enrichment.py` | Phase 3 — update `SEMANTIC_SYNTHESIS_SYSTEM_PROMPT` with structured refuse pathway |
| `backend/tests/test_requirement_enrichment.py` | New tests for evidence grounding validation, placeholder detection, refuse pathway, and post-reconstruction guard |

## Test Plan

1. **Evidence grounding validation test:** Synthesize a relation with
   title "Foo bar baz" and evidence text "something else entirely" →
   validation should reject.

2. **Evidence grounding positive test:** Synthesize a relation with
   title "Diamant ATR" and evidence text "Diamant ATR sampling
   procedure" → validation should pass.

3. **Placeholder id detection test:** Relation with `id =
   "is_about_activity"` → should be detected as placeholder.

4. **Structured refuse test:** Mock LLM returns
   `{"evidence_insufficient": true, "reason": "..."}` → synthesis
   should be rejected with the reason.

5. **Post-reconstruction guard test:** Draft has ungrounded
   `evaluated_activity` after reconstruction → guard should set
   `partial` with specific rationale.

6. **Cleanup test:** Draft has ungrounded `is_about_activity/0` →
   cleanup should remove it.

7. **No-regression test:** Run full test suite (`uv run pytest tests/ -q
   --deselect tests/test_initial_context_extraction.py::...`) — all 468
   tests must pass.

8. **End-to-end test:** Re-run profile_draft for `e5d41a78` and verify:
   - `dataset_subject_evaluation_distinction` is either `fulfilled`
     (if grounded) or `partial` with honest rationale
   - No ungrounded relations in the final draft
   - All other 12 requirements remain fulfilled
   - Semantic score ≥ 95%

## Constraints

- **No domain-specific terms** in prompts, code, or guards.  All checks
  must be generic (evidence text overlap, id format, description
  patterns).
- **Preserve evidence-backed relations.** The cleanup must not remove
  relations that were present in the initial extraction, even if their
  evidence grounding is weak.  Only remove relations that were
  synthesized by the reconstruction pipeline and lack grounding.
- **No new LLM calls.** The evidence grounding check is a deterministic
  backend operation (text overlap), not an LLM call.
- **Keep the `SEMANTIC_SYNTHESIS_SYSTEM_PROMPT` change minimal.** Only
  add the structured refuse pathway; do not restructure the prompt.

## Context for the Worker

- **Working directory:** `backend/`
- **Test command:** `uv run pytest tests/test_requirement_enrichment.py tests/test_extraction_agents.py -x -q`
- **Full test command:** `uv run pytest tests/ -q --deselect tests/test_initial_context_extraction.py::WorkflowServiceWorkflowTests::test_grounding_target_uses_manual_interim_profile_and_writes_result`
- **Profile run command:** `curl -s -X POST http://localhost:8000/api/v1/extraction/stages/profile -H "Content-Type: application/json" -d '{"data_package_id": "e5d41a78", "profile_identifier": "dcat-ap-plus", "force_rebuild": true, "chunking_strategy": "fixed_tokens", "chat_model": "rnj-1:8b-cloud"}'`
- **Progress check:** `curl -s "http://localhost:8000/api/v1/extraction/stages/evidence/e5d41a78/progress?chunking_strategy=fixed_tokens&chat_model=rnj-1:8b-cloud"`
- **Draft location:** `backend/.runtime/output/e5d41a78/profile_draft/fixed_tokens/rnj-1_8b-cloud/generated_reconstructed_draft.json`
- **Report location:** `backend/.runtime/output/e5d41a78/profile_draft/fixed_tokens/rnj-1_8b-cloud/requirement_report.json`
- **Source data:** JCAMP-DX files in `backend/.runtime/output/e5d41a78/chunks/fixed_tokens/` (see `29f10.json` for `SG-V4050.dx`, `702c5.json` for `SG-V4050.edit.jdx`)
- **Git:** Previous commit `041e92d`. Current uncommitted changes are in `projection_service.py`, `requirement_enrichment.py`, `evidence_enrichment.py`, `PROTOTYPE_STATUS.md`, `test_requirement_enrichment.py`.
- **Pre-existing test failure (acceptable):** `tests/test_initial_context_extraction.py::WorkflowServiceWorkflowTests::test_grounding_target_uses_manual_interim_profile_and_writes_result` — VocabularyCandidateSelection mock issue, unrelated.

## Expected Outcome

| Requirement | Before | After |
|---|---|---|
| `dataset_subject_evaluation_distinction` | `partial` — "Semantic evaluator returned partial without an explanation" | `fulfilled` or `partial` with specific rationale naming the ungrounded relation |
| `is_about_activity` in draft | `{"id": "is_about_activity", "title": "Infrared spectrum analysis"}` (placeholder) | Removed if ungrounded, or grounded with evidence-backed id/title |
| `evaluated_activity` in draft | `{"id": "activity:IR_spectrum_analysis", ...}` (synthesized) | Removed if ungrounded, or grounded with evidence-backed id/title |
| Semantic score | 96.1% | ≥ 95% (may drop slightly if ungrounded relations are removed, but score is honest) |