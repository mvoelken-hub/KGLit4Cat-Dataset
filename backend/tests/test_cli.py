from pathlib import Path

import pytest

from app import cli


def test_command_line_regex_matches_windows_uvicorn_executable() -> None:
    command_line = (
        '"C:\\repo\\backend\\.venv\\Scripts\\uvicorn.exe" '
        "app.main:fastapi_app --host 127.0.0.1 --port 8000 --reload"
    )

    assert cli._command_line_matches(command_line, r"\bapp\.main:fastapi_app\b", regex=True)


def test_command_line_regex_matches_vite_node_launcher() -> None:
    command_line = (
        '"node" "C:\\repo\\frontend\\node_modules\\.bin\\..\\vite\\bin\\vite.js" '
        "--host 127.0.0.1"
    )

    assert cli._command_line_matches(command_line, r"\bnpm(?:\.cmd)?\s+run\s+dev\b|\bvite(?:\.js)?\b", regex=True)


def test_frontend_dependencies_require_build_binaries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "FRONTEND_DIR", tmp_path)
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)

    assert not cli._frontend_dependencies_installed()

    (bin_dir / "tsc").touch()
    assert not cli._frontend_dependencies_installed()

    (bin_dir / "vite").touch()
    assert cli._frontend_dependencies_installed()


def test_resolve_workflow_datasets_supports_exact_glob_absolute_and_dedup(monkeypatch, tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    first = datasets_dir / "first.zip"
    second = datasets_dir / "second.zip"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    monkeypatch.setattr(cli, "DATASETS_DIR", datasets_dir)

    resolved = cli._resolve_workflow_datasets(
        ["first.zip", "*.zip", str(second)],
        all_datasets=False,
    )

    assert resolved == [first.resolve(), second.resolve()]


def test_resolve_workflow_datasets_all(monkeypatch, tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    first = datasets_dir / "a.zip"
    second = datasets_dir / "b.zip"
    ignored = datasets_dir / "readme.txt"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    ignored.write_text("nope", encoding="utf-8")
    monkeypatch.setattr(cli, "DATASETS_DIR", datasets_dir)

    assert cli._resolve_workflow_datasets([], all_datasets=True) == [
        first.resolve(),
        second.resolve(),
    ]


def test_resolve_workflow_datasets_missing(monkeypatch, tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    monkeypatch.setattr(cli, "DATASETS_DIR", datasets_dir)

    with pytest.raises(cli.typer.BadParameter, match="Dataset not found"):
        cli._resolve_workflow_datasets(["missing.zip"], all_datasets=False)


def test_build_multipart_file_body_contains_zip_file(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.zip"
    dataset.write_bytes(b"zip-bytes")

    body, content_type = cli._build_multipart_file_body(
        "file",
        dataset,
        boundary="BOUNDARY",
    )

    assert content_type == "multipart/form-data; boundary=BOUNDARY"
    assert b'name="file"; filename="sample.zip"' in body
    assert b"Content-Type: application/zip" in body
    assert b"zip-bytes" in body
    assert body.endswith(b"--BOUNDARY--\r\n")


def test_poll_status_returns_completed(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_get_json_url", lambda _url, timeout=30: (200, {"status": "completed"}, None))

    assert cli._poll_status("http://example.test", label="stage", poll_interval=0.5, timeout=1) == {
        "status": "completed"
    }


def test_poll_status_raises_on_crashed(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_get_json_url", lambda _url, timeout=30: (200, {"status": "crashed"}, None))

    with pytest.raises(RuntimeError, match="crashed"):
        cli._poll_status("http://example.test", label="stage", poll_interval=0.5, timeout=1)
