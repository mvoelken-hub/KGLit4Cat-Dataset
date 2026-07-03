from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = REPO_ROOT / "data" / "datasets" / "thesis_1H_NMR_comparison"
DEFAULT_RESULTS_DIR = DEFAULT_DATASETS_DIR / "results"
DEFAULT_RUNTIME_OUTPUT_DIR = REPO_ROOT / "backend" / ".runtime" / "output"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PROFILE_IDENTIFIER = "dcat-ap-plus"

STAGE_ORDER = [
    "upload",
    "runtime_model_switch",
    "orientation",
    "chunking",
    "evidence_context",
    "profile_construction",
    "grounding",
    "artifact_export",
]


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    file_name: str
    expected_package_id: str


@dataclass(frozen=True)
class RunConfig:
    config_id: str
    chat_model: str
    chunking_strategy: str
    generation_seed: int = 42
    fixed_tokens_per_chunk: int = 1024
    min_tokens_per_chunk: int = 128
    max_tokens_per_chunk: int = 1024
    semantic_chunking_threshold: float = 95.0
    buffer_window_size: int = 1

    @property
    def safe_model(self) -> str:
        return safe_path_component(self.chat_model)

    @property
    def safe_strategy(self) -> str:
        return safe_path_component(self.chunking_strategy)


@dataclass
class PlannedRun:
    dataset: DatasetSpec
    config: RunConfig
    result_dir: Path


@dataclass
class StageRecord:
    name: str
    endpoint: str
    request: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    final_status: str | None = None
    error: str | None = None


DATASETS = [
    DatasetSpec(
        slug="1H_NMR_raw_bruker_10",
        file_name="1H_NMR_raw_bruker_10.zip",
        expected_package_id="2e38a6a6",
    ),
    DatasetSpec(
        slug="1H_NMR_chemspectra_export",
        file_name="1H_NMR_chemspectra_export.zip",
        expected_package_id="7ac2f7cf",
    ),
]

RUN_CONFIGS = [
    RunConfig(
        config_id="fixed_1024__gemma3_12b_cloud",
        chat_model="gemma3:12b-cloud",
        chunking_strategy="fixed_tokens",
        fixed_tokens_per_chunk=1024,
        min_tokens_per_chunk=128,
        max_tokens_per_chunk=1024,
    ),
    RunConfig(
        config_id="fixed_128__gemma3_12b_cloud",
        chat_model="gemma3:12b-cloud",
        chunking_strategy="fixed_tokens",
        fixed_tokens_per_chunk=128,
        min_tokens_per_chunk=128,
        max_tokens_per_chunk=128,
    ),
    RunConfig(
        config_id="semantic_1024_128_t95_bw2__gemma3_12b_cloud",
        chat_model="gemma3:12b-cloud",
        chunking_strategy="semantic",
        min_tokens_per_chunk=128,
        max_tokens_per_chunk=1024,
        semantic_chunking_threshold=95.0,
        buffer_window_size=2,
    ),
    RunConfig(
        config_id="fixed_1024__gemma3_27b_cloud",
        chat_model="gemma3:27b-cloud",
        chunking_strategy="fixed_tokens",
        fixed_tokens_per_chunk=1024,
        min_tokens_per_chunk=128,
        max_tokens_per_chunk=1024,
    ),
]


def safe_path_component(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_batch_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_plan(
    *,
    batch_timestamp: str,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    only_dataset: str | None = None,
    only_config: str | None = None,
) -> list[PlannedRun]:
    datasets = [
        dataset
        for dataset in DATASETS
        if only_dataset is None
        or only_dataset in {dataset.slug, dataset.file_name, dataset.expected_package_id}
    ]
    configs = [
        config
        for config in RUN_CONFIGS
        if only_config is None or only_config == config.config_id
    ]
    return [
        PlannedRun(
            dataset=dataset,
            config=config,
            result_dir=results_dir / batch_timestamp / dataset.slug / config.config_id,
        )
        for dataset in datasets
        for config in configs
    ]


def stage_names() -> list[str]:
    return list(STAGE_ORDER)


class ApiError(RuntimeError):
    pass


class SimoneApiClient:
    def __init__(self, base_url: str, timeout_seconds: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get(self, path: str, query: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, query=query)

    def post(
        self,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        return self._request("POST", path, query=query, json_body=json_body)

    def patch(self, path: str, *, json_body: dict[str, Any]) -> Any:
        return self._request("PATCH", path, json_body=json_body)

    def upload_zip(self, path: str, zip_path: Path) -> Any:
        boundary = f"----SIMONEBatch{int(time.time() * 1000)}"
        body = self._multipart_body(boundary=boundary, field_name="file", file_path=zip_path)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        return self._request("POST", path, body=body, headers=headers)

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            encoded = urllib.parse.urlencode(
                {key: value for key, value in query.items() if value is not None}
            )
            url = f"{url}?{encoded}"
        request_headers = dict(headers or {})
        data = body
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                if not payload:
                    return None
                return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"{method} {url} failed: {exc}") from exc

    @staticmethod
    def _multipart_body(*, boundary: str, field_name: str, file_path: Path) -> bytes:
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/zip"
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        return header + file_path.read_bytes() + footer


class BatchRunner:
    def __init__(
        self,
        *,
        client: SimoneApiClient,
        datasets_dir: Path,
        runtime_output_dir: Path,
        profile_identifier: str,
        poll_interval_seconds: float,
        stage_timeout_seconds: int,
        overwrite: bool,
    ):
        self.client = client
        self.datasets_dir = datasets_dir
        self.runtime_output_dir = runtime_output_dir
        self.profile_identifier = profile_identifier
        self.poll_interval_seconds = poll_interval_seconds
        self.stage_timeout_seconds = stage_timeout_seconds
        self.overwrite = overwrite

    def run_all(self, plan: list[PlannedRun]) -> int:
        failures = 0
        previous_runtime = self._current_runtime_config()
        try:
            for planned in plan:
                try:
                    self.run_one(planned)
                except Exception as exc:
                    failures += 1
                    print(f"[FAILED] {planned.dataset.slug} / {planned.config.config_id}: {exc}", file=sys.stderr)
        finally:
            restore_patch = {
                key: previous_runtime[key]
                for key in ("chat_model", "generation_seed")
                if previous_runtime.get(key) is not None
            }
            if restore_patch:
                try:
                    self.client.patch(
                        "/api/v1/ollama-config/runtime",
                        json_body=restore_patch,
                    )
                except Exception as exc:
                    print(f"[WARN] Could not restore runtime config {restore_patch}: {exc}", file=sys.stderr)
        return 1 if failures else 0

    def run_one(self, planned: PlannedRun) -> None:
        result_dir = planned.result_dir
        if result_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"Result directory already exists: {result_dir}")
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._base_manifest(planned)
        self._write_json(result_dir / "run_manifest.json", manifest)
        source_path = self.datasets_dir / planned.dataset.file_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        try:
            self._run_stage(
                manifest,
                result_dir,
                "upload",
                "POST /api/v1/datasources",
                {"file": planned.dataset.file_name},
                lambda: self._upload_dataset(planned, source_path),
            )
            self._run_stage(
                manifest,
                result_dir,
                "runtime_model_switch",
                "PATCH /api/v1/ollama-config/runtime",
                {
                    "chat_model": planned.config.chat_model,
                    "generation_seed": planned.config.generation_seed,
                },
                lambda: self.client.patch(
                    "/api/v1/ollama-config/runtime",
                    json_body={
                        "chat_model": planned.config.chat_model,
                        "generation_seed": planned.config.generation_seed,
                    },
                ),
            )
            self._run_stage(
                manifest,
                result_dir,
                "orientation",
                f"POST /api/v1/extraction/stages/orientation/{planned.dataset.expected_package_id}",
                {"force_rerun": True},
                lambda: self._start_and_poll_orientation(planned),
            )
            self._run_stage(
                manifest,
                result_dir,
                "chunking",
                "POST /api/v1/datasources/chunk",
                self._chunk_query(planned),
                lambda: self._start_and_poll_chunking(planned),
            )
            self._run_stage(
                manifest,
                result_dir,
                "evidence_context",
                "POST /api/v1/extraction/stages/evidence",
                self._evidence_request(planned),
                lambda: self._start_and_poll_evidence(planned),
            )
            self._run_stage(
                manifest,
                result_dir,
                "profile_construction",
                "POST /api/v1/extraction/stages/profile",
                self._profile_request(planned),
                lambda: self._start_and_poll_profile(planned),
            )
            self._run_stage(
                manifest,
                result_dir,
                "grounding",
                f"POST /api/v1/extraction/stages/grounding/{planned.dataset.expected_package_id}/run",
                self._branch_query(planned),
                lambda: self._start_and_poll_grounding(planned),
            )
            manifest["status"] = "completed"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            raise
        finally:
            self._run_stage(
                manifest,
                result_dir,
                "artifact_export",
                "local runtime artifact copy",
                {"runtime_output_dir": str(self.runtime_output_dir)},
                lambda: self._export_artifacts(planned, result_dir, manifest),
                swallow_errors=True,
            )
            self._write_json(result_dir / "run_manifest.json", manifest)

    def _run_stage(
        self,
        manifest: dict[str, Any],
        result_dir: Path,
        name: str,
        endpoint: str,
        request: dict[str, Any],
        action: Any,
        *,
        swallow_errors: bool = False,
    ) -> Any:
        record = StageRecord(name=name, endpoint=endpoint, request=request, started_at=utc_now())
        start = time.monotonic()
        try:
            result = action()
            record.final_status = status_from_payload(result) or "completed"
            return result
        except Exception as exc:
            record.final_status = "failed"
            record.error = str(exc)
            if not swallow_errors:
                raise
            print(f"[WARN] {name} failed: {exc}", file=sys.stderr)
            return None
        finally:
            record.finished_at = utc_now()
            record.duration_seconds = round(time.monotonic() - start, 3)
            manifest["stages"][name] = asdict(record)
            self._write_json(result_dir / "run_manifest.json", manifest)

    def _upload_dataset(self, planned: PlannedRun, source_path: Path) -> Any:
        response = self.client.upload_zip("/api/v1/datasources", source_path)
        package_id = response.get("id") if isinstance(response, dict) else None
        if package_id != planned.dataset.expected_package_id:
            raise RuntimeError(
                f"Uploaded package id {package_id!r} did not match expected {planned.dataset.expected_package_id!r}"
            )
        return response

    def _start_and_poll_orientation(self, planned: PlannedRun) -> Any:
        self.client.post(
            f"/api/v1/extraction/stages/orientation/{planned.dataset.expected_package_id}",
            json_body={"force_rerun": True},
        )
        return self._poll(
            f"/api/v1/extraction/stages/orientation/{planned.dataset.expected_package_id}/progress",
            query=None,
        )

    def _start_and_poll_chunking(self, planned: PlannedRun) -> Any:
        self.client.post("/api/v1/datasources/chunk", query=self._chunk_query(planned))
        return self._poll(
            f"/api/v1/datasources/{planned.dataset.expected_package_id}/chunks/status",
            query={"chunking_strategy": planned.config.chunking_strategy},
            status_path=("status",),
            success_extra=lambda payload: bool(payload.get("has_chunks")),
        )

    def _start_and_poll_evidence(self, planned: PlannedRun) -> Any:
        self.client.post("/api/v1/extraction/stages/evidence", json_body=self._evidence_request(planned))
        return self._poll(
            f"/api/v1/extraction/stages/evidence/{planned.dataset.expected_package_id}/progress",
            query=self._branch_query(planned),
        )

    def _start_and_poll_profile(self, planned: PlannedRun) -> Any:
        self.client.post("/api/v1/extraction/stages/profile", json_body=self._profile_request(planned))
        return self._poll(
            f"/api/v1/extraction/stages/evidence/{planned.dataset.expected_package_id}/progress",
            query=self._branch_query(planned),
        )

    def _start_and_poll_grounding(self, planned: PlannedRun) -> Any:
        self.client.post(
            f"/api/v1/extraction/stages/grounding/{planned.dataset.expected_package_id}/run",
            query=self._branch_query(planned),
        )
        return self._poll(
            f"/api/v1/extraction/stages/grounding/{planned.dataset.expected_package_id}/progress",
            query=self._branch_query(planned),
        )

    def _poll(
        self,
        path: str,
        *,
        query: dict[str, Any] | None,
        status_path: tuple[str, ...] = ("status",),
        success_extra: Any | None = None,
    ) -> Any:
        deadline = time.monotonic() + self.stage_timeout_seconds
        last_payload: Any = None
        while time.monotonic() < deadline:
            payload = self.client.get(path, query=query)
            last_payload = payload
            status = nested_get(payload, status_path)
            if status == "completed" and (success_extra is None or success_extra(payload)):
                return payload
            if status in {"crashed", "cancelled"}:
                raise RuntimeError(f"Stage ended with status {status}: {payload}")
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out polling {path}: last payload={last_payload}")

    def _export_artifacts(
        self,
        planned: PlannedRun,
        result_dir: Path,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        source_path = self.datasets_dir / planned.dataset.file_name
        shutil.copy2(source_path, result_dir / "source.zip")
        self._write_json(result_dir / "config.json", self._config_payload(planned))

        token_usage = self.client.get(
            f"/api/v1/extraction/workflows/{planned.dataset.expected_package_id}/token-usage",
            query=self._branch_query(planned),
        )
        self._write_json(result_dir / "api_token_usage.json", token_usage)

        workflow_dir = self.runtime_output_dir / planned.dataset.expected_package_id
        runtime_target = result_dir / "runtime_artifacts"
        runtime_target.mkdir(parents=True, exist_ok=True)
        copied = []
        for source in self._runtime_sources(workflow_dir, planned.config):
            if not source.exists():
                continue
            destination = runtime_target / source.relative_to(workflow_dir)
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            copied.append(source.relative_to(workflow_dir).as_posix())

        self._copy_key_artifacts(runtime_target, result_dir, planned.config)
        manifest["copied_runtime_artifacts"] = copied
        return {"copied": copied}

    def _runtime_sources(self, workflow_dir: Path, config: RunConfig) -> list[Path]:
        return [
            workflow_dir / "artifact_index.json",
            workflow_dir / "extraction_warnings.json",
            workflow_dir / "chunks" / config.safe_strategy,
            workflow_dir / "overview" / config.safe_model,
            workflow_dir / "evidence_notes" / config.safe_strategy / config.safe_model,
            workflow_dir / "profile_draft" / config.safe_strategy / config.safe_model,
            workflow_dir / "grounding" / config.safe_strategy / config.safe_model,
            workflow_dir / "result" / config.safe_strategy / config.safe_model,
            workflow_dir / "run_state" / config.safe_strategy / config.safe_model,
        ]

    def _copy_key_artifacts(self, runtime_target: Path, result_dir: Path, config: RunConfig) -> None:
        key_files = {
            "grounded_final_draft.json": runtime_target / "grounding" / config.safe_strategy / config.safe_model / "grounded_final_draft.json",
            "generated_reconstructed_draft.json": runtime_target / "profile_draft" / config.safe_strategy / config.safe_model / "generated_reconstructed_draft.json",
            "generated_attribute_draft.json": runtime_target / "profile_draft" / config.safe_strategy / config.safe_model / "generated_attribute_draft.json",
            "generated_core_draft.json": runtime_target / "profile_draft" / config.safe_strategy / config.safe_model / "generated_core_draft.json",
            "requirement_report.json": runtime_target / "profile_draft" / config.safe_strategy / config.safe_model / "requirement_report.json",
            "vocab_queries.json": runtime_target / "grounding" / config.safe_strategy / config.safe_model / "vocab_queries.json",
            "grounded_validation.json": runtime_target / "grounding" / config.safe_strategy / config.safe_model / "grounded_validation.json",
            "token_usage.json": runtime_target / "result" / config.safe_strategy / config.safe_model / "token_usage.json",
            "extraction_run_state.json": runtime_target / "run_state" / config.safe_strategy / config.safe_model / "extraction_run_state.json",
        }
        for name, source in key_files.items():
            if source.exists():
                shutil.copy2(source, result_dir / name)

    def _base_manifest(self, planned: PlannedRun) -> dict[str, Any]:
        source_path = self.datasets_dir / planned.dataset.file_name
        return {
            "status": "running",
            "created_at": utc_now(),
            "dataset": asdict(planned.dataset),
            "source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path) if source_path.exists() else None,
            },
            "profile_identifier": self.profile_identifier,
            "config": asdict(planned.config),
            "stages": {name: None for name in STAGE_ORDER},
        }

    def _config_payload(self, planned: PlannedRun) -> dict[str, Any]:
        return {
            "dataset": asdict(planned.dataset),
            "profile_identifier": self.profile_identifier,
            "run_config": asdict(planned.config),
            "stage_order": STAGE_ORDER,
        }

    def _chunk_query(self, planned: PlannedRun) -> dict[str, Any]:
        config = planned.config
        return {
            "id": planned.dataset.expected_package_id,
            "buffer_window_size": config.buffer_window_size,
            "semantic_chunking_threshold": config.semantic_chunking_threshold,
            "replace_existing_chunks": "true",
            "chunking_strategy": config.chunking_strategy,
            "fixed_tokens_per_chunk": config.fixed_tokens_per_chunk,
            "min_tokens_per_chunk": config.min_tokens_per_chunk,
            "max_tokens_per_chunk": config.max_tokens_per_chunk,
        }

    def _branch_query(self, planned: PlannedRun) -> dict[str, Any]:
        return {
            "chunking_strategy": planned.config.chunking_strategy,
            "chat_model": planned.config.chat_model,
        }

    def _evidence_request(self, planned: PlannedRun) -> dict[str, Any]:
        return {
            "data_package_id": planned.dataset.expected_package_id,
            "profile_identifier": self.profile_identifier,
            "resume": False,
            "force_profile_rebuild": False,
            "target_stage": "context",
            "chunking_strategy": planned.config.chunking_strategy,
            "chat_model": planned.config.chat_model,
        }

    def _profile_request(self, planned: PlannedRun) -> dict[str, Any]:
        return {
            "data_package_id": planned.dataset.expected_package_id,
            "profile_identifier": self.profile_identifier,
            "resume": False,
            "force_rebuild": True,
            "chunking_strategy": planned.config.chunking_strategy,
            "chat_model": planned.config.chat_model,
        }

    def _current_runtime_config(self) -> dict[str, Any]:
        try:
            config = self.client.get("/api/v1/ollama-config")
            runtime = config.get("runtime") if isinstance(config, dict) else None
            if not isinstance(runtime, dict):
                return {}
            return {
                "chat_model": runtime.get("chat_model"),
                "generation_seed": runtime.get("generation_seed"),
            }
        except Exception as exc:
            print(f"[WARN] Could not read current runtime config: {exc}", file=sys.stderr)
            return {}

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def nested_get(payload: Any, path: tuple[str, ...]) -> Any:
    current = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def status_from_payload(payload: Any) -> str | None:
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str):
            return status
    return None


def print_dry_run(plan: list[PlannedRun], *, base_url: str) -> None:
    payload = {
        "base_url": base_url,
        "stage_order": STAGE_ORDER,
        "runs": [
            {
                "dataset": asdict(planned.dataset),
                "config": asdict(planned.config),
                "result_dir": str(planned.result_dir),
            }
            for planned in plan
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thesis 1H-NMR SIMONE batch via public API endpoints.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--runtime-output-dir", type=Path, default=DEFAULT_RUNTIME_OUTPUT_DIR)
    parser.add_argument("--profile-identifier", default=DEFAULT_PROFILE_IDENTIFIER)
    parser.add_argument("--batch-timestamp", default=default_batch_timestamp())
    parser.add_argument("--only-dataset", help="Dataset slug, ZIP filename, or package id to run.")
    parser.add_argument("--only-config", help="Configuration id to run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--stage-timeout-minutes", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    plan = build_run_plan(
        batch_timestamp=args.batch_timestamp,
        results_dir=args.results_dir,
        only_dataset=args.only_dataset,
        only_config=args.only_config,
    )
    if not plan:
        print("No runs selected.", file=sys.stderr)
        return 2
    if args.dry_run:
        print_dry_run(plan, base_url=args.base_url)
        return 0

    client = SimoneApiClient(args.base_url)
    client.get("/api/v1/health")
    runner = BatchRunner(
        client=client,
        datasets_dir=args.datasets_dir,
        runtime_output_dir=args.runtime_output_dir,
        profile_identifier=args.profile_identifier,
        poll_interval_seconds=args.poll_interval_seconds,
        stage_timeout_seconds=args.stage_timeout_minutes * 60,
        overwrite=args.overwrite,
    )
    return runner.run_all(plan)


if __name__ == "__main__":
    raise SystemExit(main())
