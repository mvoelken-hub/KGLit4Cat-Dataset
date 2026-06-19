from __future__ import annotations

import csv
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from app.domain.datasources import DataPackage
from app.evaluation.models import (
    EvaluationRunManifest,
    ManualReferenceBaselineReport,
    PartialEvaluationReport,
    load_reference,
)
from app.evaluation.scoring import (
    evaluate_extraction_result,
    load_result_artifacts,
    write_evaluation_report,
)


DEFAULT_EVALUATION_DATASETS = [
    "13C-Gel-NMR-13C-Gel-NMR.zip",
    "1H_NMR-1H_NMR.zip",
    "IR-IR.zip",
    "new-SCR252_A.zip",
    "13C_Gel-NMR-13C_Gel-NMR(1).zip",
]


def package_id_for_dataset(dataset_path: Path) -> str:
    package = DataPackage.from_bytes(
        BytesIO(dataset_path.read_bytes()),
        file_name=dataset_path.name,
    )
    return package.id


def _api_url(api_base: str, path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    base = api_base.rstrip("/")
    path = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"
    parsed = urllib.parse.urlparse(base)
    if parsed.path and path.startswith(f"{parsed.path.rstrip('/')}/"):
        return urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, path, "", "", "")
        )
    return f"{base}{path}"


def score_reference_directory(
    *,
    reference_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    results_dir: Path,
) -> list[Path]:
    report_paths: list[Path] = []
    partial_report_paths: list[Path] = []
    missing_outputs: list[dict[str, str]] = []
    for reference_path in sorted(reference_dir.glob("*.json")):
        reference = load_reference(reference_path)
        dataset_path = dataset_dir / reference.dataset_filename
        package_id = reference.package_id or package_id_for_dataset(dataset_path)
        workflow_dir = output_dir / package_id
        result_path = workflow_dir / "extraction_result.json"
        if not result_path.exists():
            result_candidates = sorted(
                workflow_dir.glob("*/extraction_result.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if result_candidates:
                result_path = result_candidates[0]
        if not result_path.exists():
            missing_outputs.append(
                {
                    "dataset_filename": reference.dataset_filename,
                    "package_id": package_id,
                    "missing": str(result_path),
                }
            )
            partial_report_paths.append(
                write_partial_report(
                    dataset_filename=reference.dataset_filename,
                    package_id=package_id,
                    run_dir=results_dir / package_id,
                    outcome="missing_output",
                    message=f"Completed extraction output is missing: {result_path}",
                )
            )
            continue
        result, state = load_result_artifacts(output_dir=output_dir, package_id=package_id)
        manifest = _load_manifest(results_dir / package_id / "manifest.json")
        report = evaluate_extraction_result(
            reference=reference.model_copy(update={"package_id": package_id}),
            result=result,
            state=state,
            manifest=manifest,
            schema_valid=True,
        )
        report_path = results_dir / package_id / "evaluation_report.json"
        write_evaluation_report(report, report_path)
        report_paths.append(report_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "missing_outputs.json").write_text(
        json.dumps(missing_outputs, indent=2),
        encoding="utf-8",
    )
    write_summary_tables(report_paths, partial_report_paths, results_dir)
    write_manual_reference_baseline(reference_dir=reference_dir, results_dir=results_dir)
    return report_paths


def write_manual_reference_baseline(
    *,
    reference_dir: Path,
    results_dir: Path,
) -> Path:
    rows = []
    for reference_path in sorted(reference_dir.glob("*.json")):
        reference = load_reference(reference_path)
        rows.append(
            {
                "dataset": reference.dataset_filename,
                "relevant_files": len(reference.relevant_files),
                "expected_objects": len(reference.expected_objects),
                "expected_attributes": len(reference.expected_attributes),
                "expected_vocab_mappings": len(reference.expected_vocab_mappings),
                "required_profile_fields": len(reference.required_profile_fields),
            }
        )
    report = ManualReferenceBaselineReport(
        dataset_count=len(rows),
        expected_object_count=sum(row["expected_objects"] for row in rows),
        expected_attribute_count=sum(row["expected_attributes"] for row in rows),
        expected_vocab_mapping_count=sum(row["expected_vocab_mappings"] for row in rows),
        relevant_file_count=sum(row["relevant_files"] for row in rows),
        required_profile_field_count=sum(row["required_profile_fields"] for row in rows),
        per_dataset=rows,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "manual_reference_baseline.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (results_dir / "manual_reference_baseline.md").write_text(
        _manual_reference_baseline_markdown(report),
        encoding="utf-8",
    )
    return json_path


def write_summary_tables(
    report_paths: list[Path],
    partial_report_paths: list[Path],
    results_dir: Path,
) -> None:
    rows = []
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        object_f1_values = [
            metrics["f1"]
            for metrics in payload["object_metrics"].values()
            if metrics["true_positives"] or metrics["false_positives"] or metrics["false_negatives"]
        ]
        rows.append(
            {
                "dataset": payload["reference"]["dataset_filename"],
                "package_id": payload["reference"].get("package_id") or "",
                "schema_valid": payload["schema_valid"],
                "file_ranking_top1_hit": payload["file_ranking_top1_hit"],
                "file_ranking_top3_recall": payload["file_ranking_top3_recall"],
                "object_macro_f1": round(sum(object_f1_values) / len(object_f1_values), 4) if object_f1_values else 0.0,
                "attribute_f1": payload["attribute_metrics"]["f1"],
                "vocab_mapping_f1": payload["vocab_mapping_metrics"]["f1"],
                "source_trace_coverage": payload["source_trace_coverage"],
                "profile_field_coverage": payload["required_profile_field_coverage"],
                "warnings_count": payload["warnings_count"],
            }
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "evaluation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    md_path = results_dir / "evaluation_summary.md"
    md_path.write_text(_summary_markdown(rows), encoding="utf-8")
    write_partial_summary(partial_report_paths, results_dir)


def write_partial_report(
    *,
    dataset_filename: str,
    package_id: str,
    run_dir: Path,
    outcome: str,
    message: str,
) -> Path:
    manifest = _load_manifest(run_dir / "manifest.json")
    progress = _load_json_or_none(run_dir / "latest_progress.json") or {}
    progress_body = progress.get("progress") if isinstance(progress.get("progress"), dict) else {}
    total_chunks = _safe_int(progress_body.get("total_chunks"))
    processed_chunks = _safe_int(progress_body.get("processed_chunks"))
    warnings = progress_body.get("warnings")
    report = PartialEvaluationReport(
        dataset_filename=dataset_filename,
        package_id=package_id,
        manifest=manifest,
        outcome=outcome,
        status=str(progress.get("status") or "unknown"),
        stage=progress_body.get("stage"),
        processed_chunks=processed_chunks,
        total_chunks=total_chunks,
        completion_fraction=round(processed_chunks / total_chunks, 4) if total_chunks else 0.0,
        warnings_count=len(warnings) if isinstance(warnings, list) else 0,
        message=message,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "partial_report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_partial_summary(partial_report_paths: list[Path], results_dir: Path) -> None:
    rows = []
    for path in partial_report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": payload["dataset_filename"],
                "package_id": payload["package_id"],
                "outcome": payload["outcome"],
                "status": payload["status"],
                "stage": payload.get("stage") or "",
                "processed_chunks": payload["processed_chunks"],
                "total_chunks": payload["total_chunks"],
                "completion_fraction": payload["completion_fraction"],
                "warnings_count": payload["warnings_count"],
            }
        )
    csv_path = results_dir / "partial_evaluation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "partial_evaluation_summary.md").write_text(
        _partial_summary_markdown(rows),
        encoding="utf-8",
    )


def submit_complete_workflow(
    *,
    api_base: str,
    dataset_path: Path,
    profile_identifier: str,
    qualitative_vocab_identifiers: list[str],
    results_dir: Path,
    poll_interval_seconds: float,
    timeout_seconds: float,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    max_context_length: int | None = None,
    buffer_window_size: int = 1,
    semantic_chunking_threshold: int = 95,
    chunking_strategy: str = "semantic",
    fixed_tokens_per_chunk: int = 1024,
    min_tokens_per_chunk: int = 128,
    max_tokens_per_chunk: int = 1024,
    replace_existing_chunks: bool = False,
    resume: bool = False,
    force_rerun: bool = False,
    pause_on_timeout: bool = True,
) -> Path:
    status, payload = _post_multipart(
        f"{api_base.rstrip('/')}/extraction/workflows",
        fields={
            "profile_identifier": profile_identifier,
            "qualitative_vocab_identifiers": json.dumps(qualitative_vocab_identifiers),
            "buffer_window_size": str(buffer_window_size),
            "semantic_chunking_threshold": str(semantic_chunking_threshold),
            "chunking_strategy": chunking_strategy,
            "fixed_tokens_per_chunk": str(fixed_tokens_per_chunk),
            "min_tokens_per_chunk": str(min_tokens_per_chunk),
            "max_tokens_per_chunk": str(max_tokens_per_chunk),
            "replace_existing_chunks": str(replace_existing_chunks).lower(),
            "resume": str(resume).lower(),
            "force_rerun": str(force_rerun).lower(),
        },
        file_field="file",
        file_path=dataset_path,
    )
    if status < 200 or status >= 300:
        raise RuntimeError(f"Complete workflow submission failed for {dataset_path.name}: HTTP {status} {payload}")
    package_id = payload["data_package"]["id"]
    run_dir = results_dir / package_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "submission_response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = EvaluationRunManifest(
        dataset_filename=dataset_path.name,
        package_id=package_id,
        profile_identifier=profile_identifier,
        chat_model=chat_model,
        embedding_model=embedding_model,
        max_context_length=max_context_length,
        qualitative_vocab_identifiers=qualitative_vocab_identifiers,
        chunking={
            "buffer_window_size": buffer_window_size,
            "semantic_chunking_threshold": semantic_chunking_threshold,
            "chunking_strategy": chunking_strategy,
            "fixed_tokens_per_chunk": fixed_tokens_per_chunk,
            "min_tokens_per_chunk": min_tokens_per_chunk,
            "max_tokens_per_chunk": max_tokens_per_chunk,
            "replace_existing_chunks": replace_existing_chunks,
            "resume": resume,
            "force_rerun": force_rerun,
        },
        git_commit=_git_commit(),
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result_url = f"{api_base.rstrip('/')}/extraction/results/{package_id}"
    progress_path = payload.get("progress_url") or f"/extraction/workflows/{package_id}/progress"
    progress_url = _api_url(api_base, str(progress_path))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        progress_status, progress_payload = _get_json(progress_url)
        if progress_status and 200 <= progress_status < 300:
            (run_dir / "latest_progress.json").write_text(
                json.dumps(progress_payload, indent=2),
                encoding="utf-8",
            )
            if progress_payload.get("status") == "crashed":
                write_partial_report(
                    dataset_filename=dataset_path.name,
                    package_id=package_id,
                    run_dir=run_dir,
                    outcome="crashed",
                    message=f"Extraction crashed for {dataset_path.name}.",
                )
                raise RuntimeError(f"Extraction crashed for {dataset_path.name}: {progress_payload}")

        result_status, result_payload = _get_json(result_url)
        if result_status == 200 and result_payload is not None:
            (run_dir / "api_result.json").write_text(
                json.dumps(result_payload, indent=2),
                encoding="utf-8",
            )
            return run_dir
        time.sleep(poll_interval_seconds)
    if pause_on_timeout:
        _post_json(f"{api_base.rstrip('/')}/extraction/workflows/{package_id}/pause")
    write_partial_report(
        dataset_filename=dataset_path.name,
        package_id=package_id,
        run_dir=run_dir,
        outcome="timeout",
        message=f"Timed out after {timeout_seconds} seconds waiting for a complete workflow result.",
    )
    raise TimeoutError(f"Timed out waiting for complete workflow result for {dataset_path.name}")


def _post_multipart(
    url: str,
    *,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[int, dict[str, Any]]:
    boundary = f"simone-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        ).encode()
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def _get_json(url: str) -> tuple[int | None, dict[str, Any] | None]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return exc.code, None
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload
    except Exception:
        return None, None


def _post_json(url: str) -> tuple[int | None, dict[str, Any] | None]:
    try:
        request = urllib.request.Request(
            url,
            data=b"",
            method="POST",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload
    except Exception:
        return None, None


def _summary_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# SIMONE Evaluation Summary\n\nNo completed reports found.\n"
    headers = list(rows[0].keys())
    lines = [
        "# SIMONE Evaluation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _partial_summary_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# SIMONE Partial Evaluation Summary\n\nNo partial reports found.\n"
    headers = list(rows[0].keys())
    lines = [
        "# SIMONE Partial Evaluation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _manual_reference_baseline_markdown(report: ManualReferenceBaselineReport) -> str:
    lines = [
        "# SIMONE Manual Reference Baseline",
        "",
        "This is a coverage summary of the focused manual reference annotations. "
        "It is a target for scoring selected facts, not a complete manual extraction baseline.",
        "",
        f"- datasets: {report.dataset_count}",
        f"- relevant files: {report.relevant_file_count}",
        f"- expected objects: {report.expected_object_count}",
        f"- expected attributes: {report.expected_attribute_count}",
        f"- expected vocabulary mappings: {report.expected_vocab_mapping_count}",
        f"- required profile fields: {report.required_profile_field_count}",
        "",
        "| dataset | files | objects | attributes | vocab mappings | profile fields |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.per_dataset:
        lines.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "dataset",
                    "relevant_files",
                    "expected_objects",
                    "expected_attributes",
                    "expected_vocab_mappings",
                    "required_profile_fields",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _load_manifest(path: Path) -> EvaluationRunManifest | None:
    if not path.exists():
        return None
    return EvaluationRunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _load_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
