from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.domain.extraction import ExtractionContext, ExtractionRunResult, ExtractionRunState
from app.evaluation.models import (
    AttributeMatch,
    EvaluationReference,
    EvaluationReport,
    EvaluationRunManifest,
    MetricSummary,
    ReferenceAttribute,
    ReferenceObject,
    ReferenceVocabMapping,
)


def evaluate_extraction_result(
    *,
    reference: EvaluationReference,
    result: ExtractionRunResult,
    state: ExtractionRunState | None = None,
    manifest: EvaluationRunManifest | None = None,
    schema_valid: bool = True,
) -> EvaluationReport:
    context = result.extraction_context
    object_metrics, object_failures = _score_objects(reference.expected_objects, context)
    attribute_metrics, attribute_matches = _score_attributes(
        reference.expected_attributes,
        context,
    )
    vocab_metrics, vocab_failures = _score_vocab_mappings(
        reference.expected_vocab_mappings,
        result,
    )
    profile_coverage, profile_failures = _score_profile_fields(
        reference.required_profile_fields,
        result.document,
    )
    top1_hit, top3_recall = _score_file_ranking(
        reference.relevant_files,
        state,
    )

    traced = context.extraction_objects
    source_trace_coverage = (
        sum(1 for item in traced if item.source_text.strip()) / len(traced)
        if traced
        else 0.0
    )

    failures = [
        *object_failures,
        *vocab_failures,
        *profile_failures,
    ]
    if reference.relevant_files and not top1_hit:
        failures.append("Top-ranked file is not in the reference relevant-file set.")

    return EvaluationReport(
        reference=reference,
        manifest=manifest,
        schema_valid=schema_valid,
        file_ranking_top1_hit=top1_hit,
        file_ranking_top3_recall=top3_recall,
        object_metrics=object_metrics,
        attribute_metrics=attribute_metrics,
        source_trace_coverage=source_trace_coverage,
        vocab_mapping_metrics=vocab_metrics,
        required_profile_field_coverage=profile_coverage,
        warnings_count=len(result.warnings),
        token_usage=result.token_usage,
        failures=failures,
        attribute_matches=attribute_matches,
    )


def write_evaluation_report(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_result_artifacts(
    *,
    output_dir: Path,
    package_id: str,
) -> tuple[ExtractionRunResult, ExtractionRunState | None]:
    workflow_dir = output_dir / package_id
    result = ExtractionRunResult.model_validate_json(
        (workflow_dir / "extraction_result.json").read_text(encoding="utf-8")
    )
    state_path = workflow_dir / "extraction_run_state.json"
    state = (
        ExtractionRunState.model_validate_json(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else None
    )
    return result, state


def _score_objects(
    expected_objects: list[ReferenceObject],
    context: ExtractionContext,
) -> tuple[dict[str, MetricSummary], list[str]]:
    expected_by_type: dict[str, list[ReferenceObject]] = defaultdict(list)
    actual_by_type: dict[str, list[str]] = defaultdict(list)
    for expected in expected_objects:
        expected_by_type[expected.object_type].append(expected)
    for trace in context.extraction_objects:
        actual_by_type[trace.object_type].append(_object_text(trace.extracted_object))

    metrics: dict[str, MetricSummary] = {}
    failures: list[str] = []
    for object_type, expected_items in expected_by_type.items():
        actual_items = actual_by_type.get(object_type, [])
        matched_actual: set[int] = set()
        true_positives = 0
        for expected in expected_items:
            actual_index = _best_match_index(_reference_terms(expected), actual_items, matched_actual)
            if actual_index is None:
                if expected.required:
                    failures.append(f"Missing expected {object_type}: {expected.label}")
                continue
            matched_actual.add(actual_index)
            true_positives += 1
        false_positives = max(0, len(actual_items) - len(matched_actual))
        false_negatives = max(0, len([item for item in expected_items if item.required]) - true_positives)
        metrics[object_type] = _metric_summary(true_positives, false_positives, false_negatives)

    for object_type, actual_items in actual_by_type.items():
        if object_type not in metrics:
            metrics[object_type] = _metric_summary(0, len(actual_items), 0)
    return metrics, failures


def _score_attributes(
    expected_attributes: list[ReferenceAttribute],
    context: ExtractionContext,
) -> tuple[MetricSummary, list[AttributeMatch]]:
    actual = []
    for trace in context.extraction_objects:
        obj = trace.extracted_object
        for attr in getattr(obj, "has_quantitative_attributes", []):
            title = " ".join(
                str(part)
                for part in (attr.identifier, attr.quantity_kind)
                if part
            )
            actual.append(
                {
                    "object_type": trace.object_type,
                    "title": title,
                    "value": str(attr.value),
                    "text": f"{trace.object_type} {title} {attr.value} {attr.unit}",
                }
            )
        for attr in getattr(obj, "has_qualitative_attributes", []):
            actual.append(
                {
                    "object_type": trace.object_type,
                    "title": str(attr.title),
                    "value": str(attr.value),
                    "text": f"{trace.object_type} {attr.title} {attr.value}",
                }
            )

    matched_actual: set[int] = set()
    matches: list[AttributeMatch] = []
    true_positives = 0
    for expected in expected_attributes:
        index = _matching_attribute_index(expected, actual, matched_actual)
        if index is None:
            matches.append(AttributeMatch(expected=f"{expected.title}: {expected.value}"))
            continue
        matched_actual.add(index)
        true_positives += 1
        matches.append(
            AttributeMatch(
                expected=f"{expected.title}: {expected.value}",
                matched=actual[index]["text"],
                status="matched",
            )
        )

    false_positives = max(0, len(actual) - len(matched_actual))
    false_negatives = max(0, len([item for item in expected_attributes if item.required]) - true_positives)
    return _metric_summary(true_positives, false_positives, false_negatives), matches


def _matching_attribute_index(
    expected: ReferenceAttribute,
    actual: list[dict[str, str]],
    matched_actual: set[int],
) -> int | None:
    best_index = None
    best_score = 0.0
    title_terms = [expected.title, *expected.aliases]
    for index, candidate in enumerate(actual):
        if index in matched_actual:
            continue
        if expected.object_type and candidate["object_type"] != expected.object_type:
            continue
        title_score = max(
            (_similarity(term, candidate["title"]) for term in title_terms if term),
            default=0.0,
        )
        value_score = _similarity(expected.value, candidate["value"])
        if title_score < 0.62 or value_score < 0.62:
            continue
        score = (title_score + value_score) / 2
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _score_vocab_mappings(
    expected_mappings: list[ReferenceVocabMapping],
    result: ExtractionRunResult,
) -> tuple[MetricSummary, list[str]]:
    if not expected_mappings:
        return _metric_summary(0, 0, 0), []
    actual = _normalization_records(result)
    matched_actual: set[int] = set()
    true_positives = 0
    failures: list[str] = []
    for expected in expected_mappings:
        index = _matching_vocab_index(expected, actual, matched_actual)
        if index is None:
            failures.append(f"Missing expected vocabulary mapping: {expected.source_value}")
            continue
        matched_actual.add(index)
        true_positives += 1
    false_positives = max(0, len(actual) - len(matched_actual))
    false_negatives = len(expected_mappings) - true_positives
    return _metric_summary(true_positives, false_positives, false_negatives), failures


def _normalization_records(result: ExtractionRunResult) -> list[dict[str, Any]]:
    if result.normalization is None:
        return []
    records: list[dict[str, Any]] = []
    for item in result.normalization.quantities:
        if item.quantity_kind is not None:
            records.append(
                {
                    "mapping_type": "quantity_kind",
                    "source_value": item.quantity_kind.source_value,
                    "selected_uri": item.quantity_kind.selected_uri,
                }
            )
        if item.unit is not None:
            records.append(
                {
                    "mapping_type": "unit",
                    "source_value": item.unit.source_value,
                    "selected_uri": item.unit.selected_uri,
                }
            )
    for item in result.normalization.qualitative_attributes:
        if item.term is not None:
            records.append(
                {
                    "mapping_type": "qualitative",
                    "source_value": item.term.source_value,
                    "selected_uri": item.term.selected_uri,
                }
            )
    return records


def _matching_vocab_index(
    expected: ReferenceVocabMapping,
    actual: list[dict[str, Any]],
    matched_actual: set[int],
) -> int | None:
    for index, candidate in enumerate(actual):
        if index in matched_actual:
            continue
        if candidate.get("mapping_type") != expected.mapping_type:
            continue
        if _similarity(candidate.get("source_value", ""), expected.source_value) < 0.72:
            continue
        selected_uri = candidate.get("selected_uri")
        if expected.expected_unresolved:
            if selected_uri is None:
                return index
            continue
        if not expected.acceptable_uris or selected_uri in expected.acceptable_uris:
            return index
    return None


def _score_profile_fields(
    fields: list[str],
    document: dict[str, Any],
) -> tuple[float, list[str]]:
    if not fields:
        return 0.0, []
    hits = 0
    failures = []
    for field in fields:
        value = _value_at_path(document, field)
        if value not in (None, "", [], {}):
            hits += 1
        else:
            failures.append(f"Missing required profile field: {field}")
    return hits / len(fields), failures


def _score_file_ranking(
    relevant_files: list[str],
    state: ExtractionRunState | None,
) -> tuple[bool, float]:
    if not relevant_files or state is None:
        return False, 0.0
    ranked = [item.file_path for item in state.ranked_files]
    relevant = set(relevant_files)
    top1_hit = bool(ranked and ranked[0] in relevant)
    top3 = set(ranked[:3])
    top3_recall = len(top3 & relevant) / len(relevant)
    return top1_hit, top3_recall


def _metric_summary(tp: int, fp: int, fn: int) -> MetricSummary:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricSummary(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def _reference_terms(item: ReferenceObject) -> list[str]:
    return [item.label, *item.aliases]


def _object_text(obj: Any) -> str:
    parts = [
        getattr(obj, "identifier", ""),
        getattr(obj, "description", ""),
        getattr(obj, "type", ""),
        " ".join(getattr(obj, "keywords", []) or []),
    ]
    return " ".join(str(part) for part in parts if part)


def _best_match_index(
    expected_terms: list[str],
    actual_items: list[str],
    matched_actual: set[int],
) -> int | None:
    best_index = None
    best_score = 0.0
    for index, actual in enumerate(actual_items):
        if index in matched_actual:
            continue
        score = max((_similarity(term, actual) for term in expected_terms if term), default=0.0)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= 0.62 else None


def _similarity(a: str, b: str) -> float:
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm in b_norm or b_norm in a_norm:
        return 1.0
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens) if a_tokens or b_tokens else 0.0
    sequence = SequenceMatcher(None, a_norm, b_norm).ratio()
    return max(jaccard, sequence)


def _normalize(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[_\-./:;(),]+", " ", text)
    return re.sub(r"\s+", " ", text)


def _value_at_path(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        return None
    return current
