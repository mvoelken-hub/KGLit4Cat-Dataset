from pathlib import Path

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
