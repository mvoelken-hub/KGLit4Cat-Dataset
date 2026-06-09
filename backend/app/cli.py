import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help="Semantic Inference Module for Ontology-driven Node Extraction (SIMONE) CLI - Manage the SIMONE application stack",
    invoke_without_command=True,
    add_completion=False,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"

ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

COMPOSE_PROD = REPO_ROOT / "docker-compose.yml"
COMPOSE_GPU = REPO_ROOT / "docker-compose.gpu.yml"

API_URL = "http://127.0.0.1:8000/docs"
API_BASE = "http://127.0.0.1:8000/api/v1"
HEALTH_URL = f"{API_BASE}/health"
NEO4J_BROWSER_URL = "http://127.0.0.1:7474/browser/"
NEO4J_DATA_DIR = REPO_ROOT / "data" / "docker" / "neo4j" / "data"
NEO4J_BACKUP_DIR = REPO_ROOT / ".backups" / "neo4j"
BOOTSTRAP_VOCABS_LOG = REPO_ROOT / ".runtime" / "bootstrap-vocabs.log"
BOOTSTRAP_VOCABS_PID = REPO_ROOT / ".runtime" / "bootstrap-vocabs.pid"


def _run(
    cmd: list[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = True,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True, env=process_env)


def _check_command(name: str) -> bool:
    """Check if a command is available on PATH."""
    result = subprocess.run(["where", name] if sys.platform == "win32" else ["which", name], capture_output=True)
    return result.returncode == 0


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
            ],
            capture_output=True,
        )
        return result.returncode == 0
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _running_bootstrap_vocab_pids() -> list[int]:
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Get-CimInstance Win32_Process | "
                        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*-m app.bootstrap initial-vocabs*' } | "
                        "Select-Object ProcessId | ConvertTo-Json -Compress"
                    ),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            processes = json.loads(result.stdout)
            if isinstance(processes, dict):
                processes = [processes]
            return [int(process["ProcessId"]) for process in processes]
        except Exception:
            return []

    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    pids = []
    for line in result.stdout.splitlines():
        if "-m app.bootstrap initial-vocabs" not in line:
            continue
        pid_text = line.strip().split(maxsplit=1)[0]
        try:
            pids.append(int(pid_text))
        except ValueError:
            pass
    return pids


def _write_bootstrap_vocab_lock(pid: int) -> None:
    BOOTSTRAP_VOCABS_PID.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": pid, "started_at": datetime.now().isoformat(timespec="seconds")}
    BOOTSTRAP_VOCABS_PID.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _remove_bootstrap_vocab_lock(pid: int) -> None:
    if not BOOTSTRAP_VOCABS_PID.exists():
        return
    try:
        payload = json.loads(BOOTSTRAP_VOCABS_PID.read_text(encoding="utf-8"))
    except Exception:
        BOOTSTRAP_VOCABS_PID.unlink(missing_ok=True)
        return
    if payload.get("pid") == pid:
        BOOTSTRAP_VOCABS_PID.unlink(missing_ok=True)


def _ensure_no_bootstrap_vocab_job() -> None:
    if BOOTSTRAP_VOCABS_PID.exists():
        try:
            payload = json.loads(BOOTSTRAP_VOCABS_PID.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except Exception:
            BOOTSTRAP_VOCABS_PID.unlink(missing_ok=True)
        else:
            if _pid_is_running(pid):
                typer.echo(f"Initial vocabulary bootstrap is already running (PID {pid}).", err=True)
                typer.echo(f"Log file: {BOOTSTRAP_VOCABS_LOG.relative_to(REPO_ROOT)}", err=True)
                raise typer.Exit(1)
            BOOTSTRAP_VOCABS_PID.unlink(missing_ok=True)

    running_pids = [pid for pid in _running_bootstrap_vocab_pids() if pid != os.getpid()]
    if running_pids:
        typer.echo(f"Initial vocabulary bootstrap is already running (PID {', '.join(map(str, running_pids))}).", err=True)
        typer.echo(f"Log file: {BOOTSTRAP_VOCABS_LOG.relative_to(REPO_ROOT)}", err=True)
        raise typer.Exit(1)


def _docker_available() -> bool:
    if not _check_command("docker"):
        return False
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    return result.returncode == 0


def _docker_compose_cmd() -> list[str]:
    """Return the correct docker compose command as a list."""
    if _check_command("docker-compose"):
        return ["docker-compose"]
    return ["docker", "compose"]


def _ensure_env_file(env_file: Path, example_file: Path) -> None:
    if not env_file.exists():
        if example_file.exists():
            env_file.write_text(example_file.read_text(), encoding="utf-8")
            typer.echo(f"Created {env_file.name} from {example_file.name}")
        else:
            typer.echo(f"Warning: {example_file.name} not found. Please create {env_file.name} manually.", err=True)


def _compose_base_cmd(env_file: Path, compose_files: list[Path]) -> list[str]:
    cmd = _docker_compose_cmd()
    cmd.extend(["--env-file", str(env_file)])
    for compose_file in compose_files:
        cmd.extend(["-f", str(compose_file)])
    return cmd


def _compose_status(env_file: Path, compose_files: list[Path]) -> str:
    result = _run(
        _compose_base_cmd(env_file, compose_files) + ["ps", "--format", "table {{.Service}}\t{{.Status}}\t{{.Ports}}"],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.stdout.strip() or "(no containers found)"


def _compose_has_containers(env_file: Path, compose_files: list[Path]) -> bool:
    result = _run(
        _compose_base_cmd(env_file, compose_files) + ["ps", "-a", "--format", "{{.ID}}"],
        cwd=REPO_ROOT,
        check=False,
    )
    return bool(result.stdout.strip())


def _compose_logs_tail(env_file: Path, compose_files: list[Path], services: list[str], tail: int = 25) -> str:
    cmd = _compose_base_cmd(env_file, compose_files) + ["logs", "--tail", str(tail), *services]
    result = _run(cmd, cwd=REPO_ROOT, check=False)
    return (result.stdout + result.stderr).strip() or "(no recent logs)"


def _compose_logs(env_file: Path, compose_files: list[Path], services: list[str]) -> str:
    cmd = _compose_base_cmd(env_file, compose_files) + ["logs", *services]
    result = _run(cmd, cwd=REPO_ROOT, check=False)
    return (result.stdout + result.stderr).strip()


def _compose_exec(
    env_file: Path,
    compose_files: list[Path],
    service: str,
    command: list[str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    cmd = _compose_base_cmd(env_file, compose_files) + ["exec", "-T", service, *command]
    return _run(cmd, cwd=REPO_ROOT, check=False, capture_output=capture_output)


def _compose_stop_services(env_file: Path, compose_files: list[Path], services: list[str]) -> None:
    cmd = _compose_base_cmd(env_file, compose_files) + ["stop", *services]
    _run(cmd, cwd=REPO_ROOT, check=False)


def _compose_container_id(env_file: Path, compose_files: list[Path], service: str) -> str:
    result = _run(
        _compose_base_cmd(env_file, compose_files) + ["ps", "-q", service],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def _read_env_file(env_file: Path) -> dict[str, str]:
    values = {}
    if not env_file.exists():
        return values
    for line in env_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _write_env_value(env_file: Path, key: str, value: str) -> None:
    """Update or append a single key=value pair in a .env file, preserving comments and ordering."""
    if not env_file.exists():
        env_file.write_text(f"{key}={value}\n", encoding="utf-8")
        return
    lines = env_file.read_text(encoding="utf-8").splitlines()
    updated = False
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        existing_key = text.split("=", 1)[0].strip()
        if existing_key == key:
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        # Append a blank line separator if the file doesn't end with one
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_local_host(hostname: str | None) -> bool:
    return not hostname or hostname.strip().lower() in {"localhost", "127.0.0.1", "::1"}


def _local_service_flags(env_values: dict[str, str]) -> tuple[bool, bool]:
    return _is_local_host(env_values.get("NEO4J_HOSTNAME")), _is_local_host(env_values.get("OLLAMA_HOSTNAME"))


def _frontend_url(env_values: Optional[dict[str, str]] = None) -> str:
    values = env_values if env_values is not None else _read_env_file(ENV_FILE)
    return f"http://127.0.0.1:{values.get('FRONTEND_PORT', '3000')}"


def _neo4j_browser_url(env_values: Optional[dict[str, str]] = None) -> str:
    values = env_values if env_values is not None else _read_env_file(ENV_FILE)
    hostname = values.get("NEO4J_HOSTNAME")
    host = "127.0.0.1" if _is_local_host(hostname) else hostname
    return f"http://{host}:7474/browser/"


def _check_dev_neo4j_auth(compose_files: list[Path]) -> bool:
    env_values = _read_env_file(ENV_FILE)
    user = env_values.get("NEO4J_USER", "neo4j")
    password = env_values.get("NEO4J_PASSWORD", "")
    port = env_values.get("NEO4J_PORT", "7687")
    last_result: subprocess.CompletedProcess | None = None
    for _ in range(30):
        result = _compose_exec(
            ENV_FILE,
            compose_files,
            "neo4j",
            ["cypher-shell", "-a", f"bolt://localhost:{port}", "-u", user, "-p", password, "RETURN 1;"],
            capture_output=True,
        )
        last_result = result
        output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if result.returncode == 0:
            return True
        if any(phrase in output for phrase in ("unauthorized", "authentication failure", "incorrect authentication")):
            break
        if "unable to connect" in output or "connection refused" in output or "serviceunavailable" in output:
            time.sleep(2)
            continue
        time.sleep(2)

    if last_result is None:
        return False

    result = last_result
    if result.returncode == 0:
        return True

    logs = _compose_logs_tail(ENV_FILE, compose_files, ["neo4j"], tail=20).lower()
    stderr = (result.stderr or "").lower()
    stdout = (result.stdout or "").lower()
    auth_failed = any(
        phrase in text
        for text in (logs, stderr, stdout)
        for phrase in ("unauthorized", "authentication failure", "incorrect authentication")
    )
    if auth_failed:
        typer.echo("Neo4j authentication failed for the credentials in .env.", err=True)
        typer.echo(f"  NEO4J_USER={user}", err=True)
        typer.echo("  NEO4J_PASSWORD=<value from .env>", err=True)
        typer.echo("", err=True)
        typer.echo("Neo4j keeps the password from the first time the data directory was initialized.", err=True)
        typer.echo("Changing NEO4J_PASSWORD later does not update the persisted database password.", err=True)
        typer.echo("Update .env to the existing Neo4j password, or run 'simone reset-neo4j'.", err=True)
    else:
        typer.echo("Neo4j did not become reachable within 60 seconds, so the local API was not started.", err=True)
        typer.echo("Run 'simone status' or check the Neo4j container logs if startup continues to fail.", err=True)
    return False


def _backup_neo4j() -> Path:
    os.environ["APP_ENV"] = "development"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    user = env_values.get("NEO4J_USER", "neo4j")
    password = env_values.get("NEO4J_PASSWORD", "")

    typer.echo("Ensuring Neo4j container is running ...")
    _run(_build_compose_cmd(ENV_FILE, [COMPOSE_PROD], action="up", services=["neo4j"], build=False), cwd=REPO_ROOT, check=False)

    backup_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.cypher"
    backup_file = NEO4J_BACKUP_DIR / backup_name
    NEO4J_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Exporting Neo4j graph to {backup_file.relative_to(REPO_ROOT)} ...")
    result = _compose_exec(
        ENV_FILE,
        [COMPOSE_PROD],
        "neo4j",
        [
            "cypher-shell",
            "-u",
            user,
            "-p",
            password,
            f"CALL apoc.export.cypher.all('{backup_name}', {{format: 'cypher-shell'}});",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        typer.echo("Backup failed. Check .env Neo4j credentials.", err=True)
        if result.stderr:
            typer.echo(result.stderr.strip(), err=True)
        raise typer.Exit(1)

    container_id = _compose_container_id(ENV_FILE, [COMPOSE_PROD], "neo4j")
    if not container_id:
        typer.echo("Backup export succeeded, but the Neo4j container could not be found.", err=True)
        raise typer.Exit(1)
    copy_result = _run(
        ["docker", "cp", f"{container_id}:/var/lib/neo4j/import/{backup_name}", str(backup_file)],
        cwd=REPO_ROOT,
        check=False,
    )
    if copy_result.returncode != 0:
        typer.echo("Backup export succeeded, but copying the backup file failed.", err=True)
        raise typer.Exit(1)
    typer.echo(f"Backup created: {backup_file.relative_to(REPO_ROOT)}")
    return backup_file


def _reset_neo4j_data_dir() -> None:
    NEO4J_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for child in NEO4J_DATA_DIR.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (NEO4J_DATA_DIR / ".gitkeep").touch()


def _get_json_url(url: str, timeout: int = 5) -> tuple[int | None, dict[str, object] | None, str | None]:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = None
        return exc.code, payload, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _post_json_url(url: str, data: dict[str, object] | None = None, method: str = "POST", timeout: int = 120) -> tuple[int | None, dict[str, object] | None, str | None]:
    """Send a JSON request to a URL and return (status_code, parsed_json, error_string)."""
    try:
        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = None
        return exc.code, payload, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _api_is_reachable() -> bool:
    """Check if the SIMONE API is reachable."""
    try:
        req = urllib.request.Request(HEALTH_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _api_get_ollama_config() -> tuple[dict[str, object] | None, str | None]:
    """Fetch the full Ollama config from the running API. Returns (config_dict, error)."""
    status, payload, error = _get_json_url(f"{API_BASE}/ollama-config", timeout=30)
    if error:
        return None, error
    if status and 200 <= status < 300 and isinstance(payload, dict):
        return payload, None
    return None, f"HTTP {status}"


def _api_patch_runtime(patch: dict[str, object]) -> tuple[dict[str, object] | None, str | None]:
    """Patch Ollama runtime config via the API. Returns (updated_config, error)."""
    status, payload, error = _post_json_url(f"{API_BASE}/ollama-config/runtime", data=patch, method="PATCH", timeout=30)
    if error:
        return None, error
    if status and 200 <= status < 300 and isinstance(payload, dict):
        return payload, None
    return None, f"HTTP {status}"


def _api_run_performance_test(patch: dict[str, object] | None = None) -> tuple[dict[str, object] | None, str | None]:
    """Run the Ollama performance test via the API. Returns (result_dict, error)."""
    status, payload, error = _post_json_url(f"{API_BASE}/ollama-config/runtime/performance-test", data=patch, method="POST", timeout=300)
    if error:
        return None, error
    if status and 200 <= status < 300 and isinstance(payload, dict):
        return payload, None
    return None, f"HTTP {status}"


def _is_ollama_chat_unauthorized(health_payload: dict[str, object] | None) -> bool:
    if not health_payload:
        return False
    checks = health_payload.get("checks")
    if not isinstance(checks, dict):
        return False
    chat_check = checks.get("ollama_chat")
    if not isinstance(chat_check, dict):
        return False
    error = chat_check.get("error")
    if not isinstance(error, dict):
        return False
    message = str(error.get("message", "")).lower()
    return chat_check.get("status") == "unavailable" and "unauthorized" in message


def _ensure_ollama_signin(env_file: Path, compose_files: list[Path], local_ollama: bool = True) -> None:
    status_code, payload, error = _get_json_url(HEALTH_URL, timeout=60)
    if error:
        typer.echo("Ollama Cloud sign-in status could not be checked yet.")
        typer.echo(f"Health check detail: {error}")
        typer.echo("The stack is running; Ollama may still be pulling, loading, or warming up models.")
        typer.echo("Run 'simone status' in a moment to see whether sign-in is required.")
        return
    if not _is_ollama_chat_unauthorized(payload):
        return
    if not local_ollama:
        typer.echo("Ollama Cloud sign-in is required, but Ollama is configured as an external service.")
        typer.echo("Run 'ollama signin' on the machine that hosts Ollama, then run 'simone status' again.")
        return

    typer.echo("")
    typer.echo("Ollama Cloud sign-in is required for the configured chat model.")
    typer.echo("Running 'ollama signin' inside the Ollama container.")
    typer.echo("If a connect link is printed, open it in your browser and complete the sign-in.")
    typer.echo("")
    result = _compose_exec(env_file, compose_files, "ollama", ["ollama", "signin"])
    if result.returncode != 0:
        typer.echo("Ollama sign-in command did not complete successfully.", err=True)
        return

    status_code, payload, error = _get_json_url(HEALTH_URL, timeout=60)
    if error:
        typer.echo(f"Ollama sign-in command finished, but health could not be rechecked: {error}", err=True)
    elif _is_ollama_chat_unauthorized(payload):
        typer.echo("Ollama chat is still unauthorized. Complete the browser sign-in, then run 'simone status' again.", err=True)
    else:
        typer.echo("Ollama Cloud sign-in looks OK.")


def _format_health_check(name: str, check: object) -> str:
    if not isinstance(check, dict):
        return f"  {name}: unknown"
    status_text = check.get("status", "unknown")
    model = check.get("model")
    details = f" ({model})" if model else ""
    error = check.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            details += f" - {message}"
    return f"  {name}: {status_text}{details}"


def _print_vocab_bootstrap_status() -> None:
    pids = sorted(set(_running_bootstrap_vocab_pids()))
    if not pids:
        if BOOTSTRAP_VOCABS_PID.exists():
            typer.echo("  Initial vocab bootstrap: not running (stale PID file)")
        else:
            typer.echo("  Initial vocab bootstrap: not running")
        return

    typer.echo(f"  Initial vocab bootstrap: running (PID {pids})")
    typer.echo(f"  Bootstrap log:          {BOOTSTRAP_VOCABS_LOG.relative_to(REPO_ROOT)}")


def _print_api_health() -> None:
    status_code, payload, error = _get_json_url(HEALTH_URL, timeout=60)
    typer.echo("")
    typer.echo("API health:")
    if error:
        typer.echo(f"  unavailable - {error}")
        return
    if not isinstance(payload, dict):
        typer.echo(f"  unavailable - unexpected response from {HEALTH_URL}")
        return
    typer.echo(f"  overall: {payload.get('status', 'unknown')} (HTTP {status_code})")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return
    typer.echo(_format_health_check("Neo4j", checks.get("neo4j")))
    typer.echo(_format_health_check("Ollama embeddings", checks.get("ollama_embedding")))
    typer.echo(_format_health_check("Ollama chat", checks.get("ollama_chat")))
    if _is_ollama_chat_unauthorized(payload):
        typer.echo("  Ollama sign-in: required. Run 'simone up' or 'simone signin-ollama' and open the connect link.")


def _print_new_log_lines(current_logs: str, previous_logs: str) -> bool:
    if not current_logs:
        return False
    if previous_logs and current_logs.startswith(previous_logs):
        new_logs = current_logs[len(previous_logs) :].strip()
    elif current_logs == previous_logs:
        new_logs = ""
    else:
        new_logs = current_logs
    if not new_logs:
        return False
    typer.echo(new_logs)
    return True


def _wait_for_url(
    url: str,
    timeout: int = 120,
    label: str = "service",
    verbose: bool = False,
    env_file: Optional[Path] = None,
    compose_files: Optional[list[Path]] = None,
    log_services: Optional[list[str]] = None,
) -> bool:
    typer.echo(f"Waiting for {label} at {url} ...")
    start = time.time()
    attempt = 0
    last_error = ""
    last_details_at = -10
    previous_logs = ""
    if verbose and env_file and compose_files and log_services:
        previous_logs = _compose_logs(env_file, compose_files, log_services)
    while time.time() - start < timeout:
        attempt += 1
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    typer.echo(f"{label} is ready.")
                    return True
        except urllib.error.HTTPError as exc:
            if 200 <= exc.code < 500:
                typer.echo(f"{label} is ready.")
                return True
            last_error = f"HTTPError: {exc.code} {exc.reason}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        elapsed = int(time.time() - start)
        if verbose and env_file and compose_files and log_services:
            current_logs = _compose_logs(env_file, compose_files, log_services)
            if _print_new_log_lines(current_logs, previous_logs):
                previous_logs = current_logs
                last_details_at = elapsed
            elif elapsed - last_details_at >= 15:
                last_details_at = elapsed
                typer.echo(f"[{elapsed:>3}s] Still waiting for {label}. Last check: {last_error}")
                typer.echo("Container status:")
                typer.echo(_compose_status(env_file, compose_files))
                typer.echo("")
        elif verbose:
            typer.echo(f"[{elapsed:>3}s] Still waiting for {label}. Last check: {last_error}")
        time.sleep(2)
    typer.echo(f"{label} was not reachable after {timeout} seconds.", err=True)
    if verbose and env_file and compose_files:
        typer.echo("Final container status:", err=True)
        typer.echo(_compose_status(env_file, compose_files), err=True)
        typer.echo("", err=True)
        typer.echo("Recent stack logs:", err=True)
        typer.echo(_compose_logs_tail(env_file, compose_files, ["api", "neo4j", "ollama", "frontend"], tail=40), err=True)
    return False


def _print_links() -> None:
    frontend_url = _frontend_url()
    neo4j_browser_url = _neo4j_browser_url()
    typer.echo("")
    typer.echo("SIMONE is starting up. You can access the services at:")
    typer.echo(f"  Frontend:    {frontend_url}")
    typer.echo(f"  API Docs:    {API_URL}")
    typer.echo(f"  Neo4j:       {neo4j_browser_url}")
    typer.echo("")


def _compose_files_for_mode(gpu: bool = True) -> list[Path]:
    files = [COMPOSE_PROD]
    if gpu:
        files.append(COMPOSE_GPU)
    return files


def _build_compose_cmd(
    env_file: Path,
    compose_files: list[Path],
    action: str,
    services: Optional[list[str]] = None,
    build: bool = False,
) -> list[str]:
    cmd = _docker_compose_cmd()
    cmd.extend(["--env-file", str(env_file)])
    for f in compose_files:
        cmd.extend(["-f", str(f)])
    cmd.append(action)
    if build and action in ("up", "create"):
        cmd.append("--build")
    if action == "up":
        cmd.append("-d")
    if services:
        cmd.extend(services)
    return cmd


def _find_pids_by_cmdline(pattern: str) -> list[int]:
    """Find Windows process IDs whose command line contains the given pattern."""
    if sys.platform != "win32":
        return []
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        processes = json.loads(result.stdout)
        if isinstance(processes, dict):
            processes = [processes]
        current_pid = os.getpid()
        pids = []
        for process in processes:
            command_line = process.get("CommandLine") or ""
            pid = process.get("ProcessId")
            if pid != current_pid and pattern in command_line:
                pids.append(int(pid))
        return pids
    except Exception:
        return []


def _kill_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        except Exception:
            pass


def _stop_bootstrap_vocab_job() -> list[int]:
    pids = sorted(set(_running_bootstrap_vocab_pids()))
    if pids:
        _kill_pids(pids)
    BOOTSTRAP_VOCABS_PID.unlink(missing_ok=True)
    return pids


def _find_pids_by_window_title(title: str) -> list[int]:
    if sys.platform != "win32":
        return []
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Process | Select-Object Id,MainWindowTitle | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        processes = json.loads(result.stdout)
        if isinstance(processes, dict):
            processes = [processes]
        return [
            int(process["Id"])
            for process in processes
            if str(process.get("MainWindowTitle") or "").startswith(title)
        ]
    except Exception:
        return []


def _kill_windows_by_title(title: str) -> None:
    pids = _find_pids_by_window_title(title)
    if pids:
        _kill_pids(pids)
        return
    try:
        subprocess.run(["taskkill", "/FI", f"WINDOWTITLE eq {title}*", "/F"], capture_output=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(ctx: typer.Context) -> None:
    """Manage the SIMONE application stack."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command()
def up(
    build: bool = typer.Option(False, "--build", help="Build images before starting containers"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show startup progress, container status, and API logs while waiting"),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Disable NVIDIA GPU reservations for Ollama (CPU-only mode)"),
) -> None:
    """Start SIMONE in production mode (API/frontend in Docker)."""
    if not _docker_available():
        typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
        raise typer.Exit(1)

    os.environ["APP_ENV"] = "production"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    local_neo4j, local_ollama = _local_service_flags(env_values)

    compose_files = _compose_files_for_mode(gpu=not no_gpu)
    services = [service for service, enabled in (("neo4j", local_neo4j), ("ollama", local_ollama)) if enabled]
    services.extend(["api", "frontend"])
    cmd = _build_compose_cmd(ENV_FILE, compose_files, action="up", services=services, build=build)

    typer.echo("Starting production stack ...")
    _run(cmd, cwd=REPO_ROOT, capture_output=not verbose)

    api_ready = _wait_for_url(
        API_URL,
        timeout=120,
        label="API",
        verbose=verbose,
        env_file=ENV_FILE,
        compose_files=compose_files,
        log_services=["api"],
    )
    if api_ready:
        typer.echo("Now checking health... (takes up to 60s)")
        _ensure_ollama_signin(ENV_FILE, compose_files, local_ollama=local_ollama)
    _print_links()


@app.command()
def dev(
    no_npm: bool = typer.Option(False, "--no-npm", help="Run the frontend in Docker instead of requiring local npm"),
    fg: bool = typer.Option(
        False,
        "-fg",
        help="Attach local API and frontend logs to this terminal instead of opening dev windows",
    ),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Disable NVIDIA GPU reservations for Ollama (CPU-only mode)"),
) -> None:
    """Start SIMONE in development mode (API locally; local services as needed)."""
    os.environ["APP_ENV"] = "development"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    local_neo4j, local_ollama = _local_service_flags(env_values)

    # Dependency sync
    if not _check_command("uv"):
        typer.echo("Error: 'uv' is not installed or not on PATH. Install it from https://docs.astral.sh/uv/", err=True)
        raise typer.Exit(1)

    if not (BACKEND_DIR / ".venv").exists():
        typer.echo("Running uv sync in backend/ ...")
        _run(["uv", "sync"], cwd=BACKEND_DIR)

    use_docker_frontend = no_npm
    if not no_npm:
        if not _check_command("npm"):
            typer.echo("")
            typer.echo("npm is not installed or not on PATH.")
            typer.echo("Install Node.js from https://nodejs.org/ if you want to work on the frontend locally.")
            typer.echo("Falling back to Docker frontend mode, same as 'simone dev --no-npm'.")
            use_docker_frontend = True
        elif not (FRONTEND_DIR / "node_modules").exists():
            typer.echo("Running npm install in frontend/ ...")
            _run(["npm", "install"], cwd=FRONTEND_DIR)
        else:
            pass

    typer.echo("")
    compose_files = _compose_files_for_mode(gpu=not no_gpu)
    services = [service for service, enabled in (("neo4j", local_neo4j), ("ollama", local_ollama)) if enabled]
    if use_docker_frontend:
        services.append("frontend")
    if services:
        if not _docker_available():
            typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
            raise typer.Exit(1)
        cmd = _build_compose_cmd(ENV_FILE, compose_files, action="up", services=services, build=True)
        container_label = " + ".join(services)
        typer.echo(f"Starting {container_label} containers ...")
        _run(cmd, cwd=REPO_ROOT)
    else:
        typer.echo("Using external Neo4j/Ollama services from .env.")

    _compose_stop_services(ENV_FILE, [COMPOSE_PROD], ["api"])
    if not use_docker_frontend:
        _compose_stop_services(ENV_FILE, [COMPOSE_PROD], ["frontend"])

    if local_neo4j and not _check_dev_neo4j_auth(compose_files):
        raise typer.Exit(1)

    typer.echo("")
    existing_api = _find_pids_by_cmdline("uvicorn app.main:fastapi_app") or _find_pids_by_window_title("SIMONE API")
    existing_frontend = _find_pids_by_cmdline("npm run dev") or _find_pids_by_cmdline("vite") or _find_pids_by_window_title("SIMONE Frontend")

    if fg:
        if existing_api:
            typer.echo("Local API is already running (PID " + str(existing_api) + "). API logs are not attached.")
            return

        frontend_process: subprocess.Popen | None = None
        frontend_log_process: subprocess.Popen | None = None
        try:
            if use_docker_frontend:
                typer.echo("Attaching Docker frontend logs in this terminal ...")
                frontend_log_process = subprocess.Popen(
                    _compose_base_cmd(ENV_FILE, compose_files) + ["logs", "-f", "frontend"],
                    cwd=str(REPO_ROOT),
                )
            elif existing_frontend:
                typer.echo("Local frontend is already running (PID " + str(existing_frontend) + "). Frontend logs are not attached.")
            else:
                npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
                typer.echo("Starting local frontend dev server in this terminal ...")
                frontend_process = subprocess.Popen(
                    [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1"],
                    cwd=str(FRONTEND_DIR),
                )

            _wait_for_url(
                _frontend_url(env_values),
                timeout=60,
                label="Frontend",
                verbose=False,
                env_file=ENV_FILE if use_docker_frontend else None,
                compose_files=compose_files if use_docker_frontend else None,
                log_services=["frontend"] if use_docker_frontend else None,
            )
            _print_links()

            typer.echo("Starting local API with hot reload in this terminal. Press Ctrl+C to stop foreground services.")
            process_env = os.environ.copy()
            process_env.update(env_values)
            process_env["APP_ENV"] = "development"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--env-file",
                    "../.env",
                    "uvicorn",
                    "app.main:fastapi_app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                    "--reload",
                ],
                cwd=str(BACKEND_DIR),
                env=process_env,
            )
            if result.returncode != 0:
                raise typer.Exit(result.returncode)
            return
        except KeyboardInterrupt:
            typer.echo("Stopping foreground dev services ...")
            raise typer.Exit(130)
        finally:
            for process in (frontend_process, frontend_log_process):
                if process is not None and process.poll() is None:
                    process.terminate()

    if not use_docker_frontend:
        # Start local frontend in a new visible terminal window
        if existing_frontend:
            typer.echo("Local frontend is already running (PID " + str(existing_frontend) + "). Skipping.")
        else:
            frontend_cmd = "npm run dev -- --host 127.0.0.1"
            typer.echo("Starting local frontend dev server ...")
            if sys.platform == "win32":
                subprocess.Popen(
                    ["cmd", "/c", "start", "SIMONE Frontend", "powershell", "-ExecutionPolicy", "Bypass", "-Command", frontend_cmd],
                    cwd=str(FRONTEND_DIR),
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                subprocess.Popen(
                    frontend_cmd,
                    cwd=str(FRONTEND_DIR),
                    shell=True,
                )
    else:
        pass

    # Start local API in a new visible terminal window
    api_cmd = (
        "$env:APP_ENV='development'; uv run --env-file ../.env uvicorn app.main:fastapi_app "
        '--host 127.0.0.1 --port 8000 --reload'
    )
    if existing_api:
        typer.echo("Local API is already running (PID " + str(existing_api) + "). Skipping.")
    else:
        typer.echo("Starting local API with hot reload ...")
        if sys.platform == "win32":
            subprocess.Popen(
                ["cmd", "/c", "start", "SIMONE API", "powershell", "-ExecutionPolicy", "Bypass", "-Command", api_cmd],
                cwd=str(BACKEND_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            subprocess.Popen(
                "APP_ENV=development uv run --env-file ../.env uvicorn app.main:fastapi_app --host 127.0.0.1 --port 8000 --reload",
                cwd=str(BACKEND_DIR),
                shell=True,
            )

    api_ready = _wait_for_url(API_URL, timeout=120, label="API", verbose=False)
    _wait_for_url(
        _frontend_url(env_values),
        timeout=60,
        label="Frontend",
        verbose=False,
        env_file=ENV_FILE if use_docker_frontend else None,
        compose_files=compose_files if use_docker_frontend else None,
        log_services=["frontend"] if use_docker_frontend else None,
    )
    if api_ready:
        typer.echo("Now checking health... (takes up to 60s)")
    _print_api_health()
    _print_links()


@app.command()
def host(
    neo4j: bool = typer.Option(False, "--neo4j", help="Start Neo4j (default: start all local services)"),
    ollama: bool = typer.Option(False, "--ollama", help="Start Ollama (default: start all local services)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show startup progress and container status"),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Disable NVIDIA GPU reservations for Ollama (CPU-only mode)"),
) -> None:
    """Start only infrastructure services (Neo4j and/or Ollama) for remote access by other SIMONE instances."""
    if not _docker_available():
        typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
        raise typer.Exit(1)

    os.environ["APP_ENV"] = "production"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    local_neo4j, local_ollama = _local_service_flags(env_values)

    # If no specific services requested, start all local services
    select_specific = neo4j or ollama
    start_neo4j = neo4j if select_specific else local_neo4j
    start_ollama = ollama if select_specific else local_ollama

    # Warn if a requested service is configured as remote
    if neo4j and not local_neo4j:
        typer.echo("Note: Neo4j is configured as a remote service in .env (NEO4J_HOSTNAME is set). Starting it anyway.")
        start_neo4j = True
    if ollama and not local_ollama:
        typer.echo("Note: Ollama is configured as a remote service in .env (OLLAMA_HOSTNAME is set). Starting it anyway.")
        start_ollama = True

    compose_services = []
    if start_neo4j:
        compose_services.append("neo4j")
    if start_ollama:
        compose_services.append("ollama")

    if not compose_services:
        typer.echo("Both Neo4j and Ollama are configured as remote services in .env.")
        typer.echo("Nothing to start locally. Use --neo4j and/or --ollama to force start, or update NEO4J_HOSTNAME and OLLAMA_HOSTNAME.")
        raise typer.Exit(0)

    compose_files = _compose_files_for_mode(gpu=not no_gpu)
    cmd = _build_compose_cmd(ENV_FILE, compose_files, action="up", services=compose_services, build=False)
    container_label = " + ".join(compose_services)
    typer.echo(f"Starting {container_label} for remote access ...")
    _run(cmd, cwd=REPO_ROOT, capture_output=not verbose)

    if start_neo4j:
        neo4j_ready = _wait_for_url(
            "http://127.0.0.1:7474",
            timeout=120,
            label="Neo4j",
            verbose=verbose,
            env_file=ENV_FILE,
            compose_files=compose_files,
            log_services=["neo4j"],
        )
        if neo4j_ready:
            typer.echo(f"Neo4j is available at bolt://127.0.0.1:{env_values.get('NEO4J_PORT', '7687')}")

    if start_ollama:
        ollama_port = env_values.get("OLLAMA_PORT", "11433")
        ollama_ready = _wait_for_url(
            f"http://127.0.0.1:{ollama_port}",
            timeout=120,
            label="Ollama",
            verbose=verbose,
            env_file=ENV_FILE,
            compose_files=compose_files,
            log_services=["ollama"],
        )
        if ollama_ready:
            typer.echo(f"Ollama is available at http://127.0.0.1:{ollama_port}")

    # Determine the machine's hostname for remote connection hints
    hostname = socket.gethostname()
    try:
        # Prefer the fully-qualified hostname; fall back to the short name
        fqdn = socket.getfqdn()
        display_host = fqdn if "." in fqdn else hostname
    except Exception:
        display_host = hostname

    typer.echo("")
    typer.echo("Infrastructure services are running. Other machines can connect by setting these values in their .env:")
    if start_neo4j:
        typer.echo(f"  NEO4J_HOSTNAME={display_host}")
        typer.echo(f"  NEO4J_PORT={env_values.get('NEO4J_PORT', '7687')}")
    if start_ollama:
        typer.echo(f"  OLLAMA_HOSTNAME={display_host}")
        typer.echo(f"  OLLAMA_PORT={env_values.get('OLLAMA_PORT', '11433')}")


@app.command("vocabs")
def vocabs(
    ctx: typer.Context,
    info: bool = typer.Option(False, "--info", help="Print the configured initial vocabularies as JSON"),
    bootstrap: bool = typer.Option(False, "--bootstrap", help="Import configured vocabularies into Neo4j and generate embeddings"),
    foreground: bool = typer.Option(False, "--foreground", help="Run bootstrap in the current terminal instead of in the background"),
    start_services: bool = typer.Option(True, "--start-services/--no-start-services", help="Start local Neo4j/Ollama containers before bootstrapping"),
) -> None:
    """Inspect or bootstrap the configured initial vocabularies."""
    if info and bootstrap:
        typer.echo("Error: choose either --info or --bootstrap, not both.", err=True)
        raise typer.Exit(1)

    if not info and not bootstrap:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)

    if info:
        _print_initial_vocab_info()
        return

    _bootstrap_vocabs(foreground=foreground, start_services=start_services)


def _print_initial_vocab_info() -> None:
    from app.core.initial_vocabs import INITIAL_VOCABS

    payload = [vocab.model_dump(mode="json") for vocab in INITIAL_VOCABS]
    typer.echo(json.dumps(payload, indent=2))


@app.command("models")
def models(
    start_services: bool = typer.Option(True, "--start-services/--no-start-services", help="Start local Ollama before interacting"),
) -> None:
    """Interactively manage Ollama models and runtime configuration."""
    os.environ["APP_ENV"] = "development"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    os.environ.update(env_values)
    os.environ["APP_ENV"] = "development"

    from app.core.config import Settings
    from app.ollama.client import OllamaClientWrapper
    from app.core.logging import logger

    model_settings = Settings()
    local_ollama = _is_local_host(env_values.get("OLLAMA_HOSTNAME"))

    if start_services and local_ollama:
        if not _docker_available():
            typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
            raise typer.Exit(1)
        typer.echo("Starting Ollama container ...")
        _run(_build_compose_cmd(ENV_FILE, [COMPOSE_PROD], action="up", services=["ollama"], build=False), cwd=REPO_ROOT)
        if not _wait_for_url(model_settings.ollama_base_url, timeout=120, label="Ollama", verbose=False):
            raise typer.Exit(1)
    elif local_ollama:
        typer.echo(f"Using local Ollama at {model_settings.ollama_base_url}.")
    else:
        typer.echo(f"Using external Ollama at {model_settings.ollama_base_url}.")

    typer.echo("Ollama model management is intended to run on the machine that hosts Ollama.")
    if local_ollama:
        typer.echo("This SIMONE environment points to local/Docker Ollama, so model operations are enabled.")
    else:
        typer.echo("This SIMONE environment points to a remote Ollama host.")
        typer.echo("Inspection is available here; run this command on the Ollama host for pull/remove/set/fit operations.")
    typer.echo(f"  Embedding model: {model_settings.ollama_embed_model}")
    typer.echo(f"  Chat model:       {model_settings.ollama_chat_model}")
    typer.echo("")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _ollama_list():
        client = OllamaClientWrapper(model_settings, logger)
        try:
            return await client.list_models()
        finally:
            await client.close()

    while True:
        # --- List all downloaded models ---
        try:
            listed = loop.run_until_complete(_ollama_list())
        except Exception as exc:
            typer.echo(f"Failed to list models: {type(exc).__name__}: {exc}", err=True)
            listed = None

        if listed and listed.models:
            typer.echo("Downloaded models:")
            typer.echo(f"  {'#':>3}  {'Model':<40} {'Size':>10} {'Family':<16} {'Params':<12} {'Quant':<8} Role")
            typer.echo(f"  {'---':>3}  {'---':<40} {'---':>10}  {'---':<16} {'---':<12} {'---':<8} ---")
            for i, m in enumerate(listed.models, 1):
                size_mb = f"{m.size.real / 1024 / 1024:.0f} MB" if m.size else ""
                family = m.details.family if m.details and m.details.family else ""
                params = m.details.parameter_size if m.details and m.details.parameter_size else ""
                quant = m.details.quantization_level if m.details and m.details.quantization_level else ""
                roles: list[str] = []
                if m.model == model_settings.ollama_embed_model:
                    roles.append("embedding")
                if m.model == model_settings.ollama_chat_model:
                    roles.append("chat")
                role_tag = ",".join(roles) if roles else ""
                typer.echo(f"  {i:>3}  {m.model:<40} {size_mb:>10} {family:<16} {params:<12} {quant:<8} {role_tag}")
        else:
            typer.echo("No models downloaded yet.")

        typer.echo("")
        typer.echo("Actions: [i] inspect  [l] loaded  [c] config  [t] test  [d] diagnostics  [p] pull  [r] remove  [s] set as  [g] fit check  [x] server  [q] quit")
        action = typer.prompt("Choose action", default="q").strip().lower()

        if action in ("q", "quit", ""):
            typer.echo("Goodbye!")
            break

        elif action in ("i", "inspect"):
            _models_action_inspect(model_settings, listed, loop)

        elif action in ("l", "loaded"):
            _models_action_loaded(model_settings, loop)

        elif action in ("c", "config"):
            _models_action_config(model_settings, env_values, loop)

        elif action in ("t", "test"):
            _models_action_test(model_settings, loop)

        elif action in ("d", "diag", "diagnostics"):
            _models_action_diagnostics(model_settings, loop)

        elif action in ("p", "pull"):
            if not local_ollama:
                typer.echo("Pull is disabled here because Ollama is remote. Run this command on the Ollama host.")
                typer.echo("")
                continue
            _models_action_pull(model_settings, loop)

        elif action in ("r", "remove"):
            if not local_ollama:
                typer.echo("Remove is disabled here because Ollama is remote. Run this command on the Ollama host.")
                typer.echo("")
                continue
            _models_action_remove(model_settings, listed, loop)

        elif action in ("s", "set"):
            if not local_ollama:
                typer.echo("Changing configured models is disabled here because Ollama is remote. Run this command on the Ollama host.")
                typer.echo("")
                continue
            _models_action_set(model_settings, listed, env_values)
            model_settings = Settings()

        elif action in ("g", "gpu", "ping"):
            if not local_ollama:
                typer.echo("Fit check is disabled here because Ollama is remote. Run this command on the Ollama host.")
                typer.echo("")
                continue
            _models_action_gpu(model_settings, loop)

        elif action in ("x", "server"):
            _models_action_server(env_values)

        else:
            typer.echo(f"Unknown action: {action}")

        typer.echo("")

    loop.close()


def _models_action_inspect(model_settings, listed, loop) -> None:
    from app.ollama.client import OllamaClientWrapper
    from app.core.logging import logger

    if not listed or not listed.models:
        typer.echo("No models to inspect.")
        return

    choice = typer.prompt("Model number to inspect", type=int)
    if choice < 1 or choice > len(listed.models):
        typer.echo(f"Invalid selection: {choice}", err=True)
        return

    model_name = listed.models[choice - 1].model
    typer.echo(f"Inspecting {model_name} ...")

    async def _run():
        client = OllamaClientWrapper(model_settings, logger)
        try:
            return await client.show_model(model_name)
        finally:
            await client.close()

    try:
        info = loop.run_until_complete(_run())
    except Exception as exc:
        typer.echo(f"Failed to inspect model: {type(exc).__name__}: {exc}", err=True)
        return

    typer.echo(f"  Model:        {model_name}")
    if info.details:
        typer.echo(f"  Family:       {info.details.family or ''}")
        typer.echo(f"  Parameter size: {info.details.parameter_size or ''}")
        typer.echo(f"  Quantization: {info.details.quantization_level or ''}")
        typer.echo(f"  Format:       {info.details.format or ''}")
    if info.capabilities:
        typer.echo(f"  Capabilities: {', '.join(info.capabilities)}")
    if info.parameters:
        typer.echo(f"  Parameters:   {info.parameters}")
    if info.template:
        typer.echo(f"  Template:      {info.template[:200]}{'...' if len(info.template) > 200 else ''}")
    if info.license:
        typer.echo(f"  License:      {info.license[:200]}{'...' if len(info.license) > 200 else ''}")
    if info.modelinfo:
        for k, v in info.modelinfo.items():
            val_str = str(v)
            typer.echo(f"  {k}: {val_str[:100]}{'...' if len(val_str) > 100 else ''}")


def _models_action_pull(model_settings, loop) -> None:
    from app.ollama.client import OllamaClientWrapper
    from app.core.logging import logger

    model_name = typer.prompt("Model name to pull (e.g. gemma3:4b)")
    if not model_name.strip():
        typer.echo("No model name provided.")
        return

    typer.echo(f"Pulling {model_name} ...")

    async def _run():
        client = OllamaClientWrapper(model_settings, logger)
        try:
            await client.pull_models([model_name.strip()])
        finally:
            await client.close()

    try:
        loop.run_until_complete(_run())
    except Exception as exc:
        typer.echo(f"Pull failed: {type(exc).__name__}: {exc}", err=True)
        return

    typer.echo(f"Pulled {model_name} successfully.")


def _models_action_remove(model_settings, listed, loop) -> None:
    from app.ollama.client import OllamaClientWrapper
    from app.core.logging import logger

    if not listed or not listed.models:
        typer.echo("No models to remove.")
        return

    choice = typer.prompt("Model number to remove", type=int)
    if choice < 1 or choice > len(listed.models):
        typer.echo(f"Invalid selection: {choice}", err=True)
        return

    model_name = listed.models[choice - 1].model
    if not typer.confirm(f"Remove {model_name}?", default=False):
        typer.echo("Cancelled.")
        return

    async def _run():
        client = OllamaClientWrapper(model_settings, logger)
        try:
            await client.delete_model(model_name)
        finally:
            await client.close()

    try:
        loop.run_until_complete(_run())
    except Exception as exc:
        typer.echo(f"Remove failed: {type(exc).__name__}: {exc}", err=True)
        return

    typer.echo(f"Removed {model_name}.")


def _models_action_set(model_settings, listed, env_values: dict[str, str]) -> None:
    if not listed or not listed.models:
        typer.echo("No models available. Pull a model first.")
        return

    choice = typer.prompt("Model number to set as configured model", type=int)
    if choice < 1 or choice > len(listed.models):
        typer.echo(f"Invalid selection: {choice}", err=True)
        return

    model_name = listed.models[choice - 1].model
    role = typer.prompt("Set as [e]mbedding or [c]hat?", default="c").strip().lower()

    if role in ("e", "embedding", "embed"):
        env_key = "OLLAMA_EMBED_MODEL"
        role_label = "embedding"
        patch_key = "embedding_model"
    elif role in ("c", "chat"):
        env_key = "OLLAMA_CHAT_MODEL"
        role_label = "chat"
        patch_key = "chat_model"
    else:
        typer.echo(f"Unknown role: {role}. Use 'embedding' or 'chat'.", err=True)
        return

    _write_env_value(ENV_FILE, env_key, model_name)
    env_values[env_key] = model_name
    os.environ[env_key] = model_name
    typer.echo(f"Set {model_name} as the {role_label} model in .env ({env_key}={model_name}).")

    # Try to hot-reload via API
    api_config, api_error = _api_get_ollama_config()
    if api_config is not None:
        patch = {patch_key: model_name}
        updated, err = _api_patch_runtime(patch)
        if err:
            typer.echo(f"  Could not apply to running API: {err}", err=True)
            typer.echo("  Restart the API for the change to take effect.")
        else:
            typer.echo(typer.style("  Applied to running API (hot-reload).", fg=typer.colors.GREEN))
            typer.echo("  Note: Runtime changes reset when the API restarts. .env values persist.")
    else:
        typer.echo("  API is not reachable. Restart the API for the change to take effect.")


def _models_action_gpu(model_settings, loop) -> None:
    from app.ollama.client import OllamaClientWrapper
    from app.core.logging import logger

    embed_model = model_settings.ollama_embed_model
    chat_model = model_settings.ollama_chat_model
    num_ctx = model_settings.max_context_length
    embed_batch_size = model_settings.embedding_batch_size
    flash_attn = model_settings.ollama_flash_attention
    kv_cache_type = model_settings.ollama_kv_cache_type
    embed_num_gpu = model_settings.ollama_embed_num_gpu

    typer.echo("Fit check - testing configured model residency without unloading existing models.")
    typer.echo(f"  MAX_CONTEXT_LENGTH={num_ctx}  (KV cache size for chat model)")
    typer.echo(f"  EMBEDDING_BATCH_SIZE={embed_batch_size}  (peak memory during embedding)")
    typer.echo(f"  OLLAMA_FLASH_ATTENTION={flash_attn}  (server-side setting on the Ollama host)")
    typer.echo(f"  OLLAMA_KV_CACHE_TYPE={kv_cache_type}  (server-side KV cache type; q8_0 and q4_0 can reduce memory)")
    embed_gpu_label = "auto (all GPU)" if embed_num_gpu == -1 else ("CPU only" if embed_num_gpu == 0 else f"{embed_num_gpu} layers")
    typer.echo(f"  OLLAMA_EMBED_NUM_GPU={embed_num_gpu}  (embedding model: {embed_gpu_label})")
    typer.echo("")

    async def _ping_all():
        client = OllamaClientWrapper(model_settings, logger)
        try:
            before = await client.list_running_models()
            before_names = {m.model for m in before.models} if before and before.models else set()
            embed_num_gpu_opt = embed_num_gpu if embed_num_gpu != -1 else None
            embed_result = await client.ping_model(embed_model, num_ctx=num_ctx, is_embedding=True, num_gpu=embed_num_gpu_opt)
            chat_result = await client.ping_model(chat_model, num_ctx=num_ctx, is_embedding=False)
            after = await client.list_running_models()
            return embed_result, chat_result, before_names, after
        except Exception:
            await client.close()
            raise
        finally:
            await client.close()

    try:
        embed_result, chat_result, before_names, running = loop.run_until_complete(_ping_all())
    except Exception as exc:
        typer.echo(f"Fit check failed: {type(exc).__name__}: {exc}", err=True)
        return

    # Report results
    for label, result in [("Embedding", embed_result), ("Chat", chat_result)]:
        model_name = result["model"]
        if result["success"]:
            load_ns = result.get("load_duration_ns")
            load_info = f"load: {load_ns / 1_000_000:.0f} ms" if load_ns else ""
            extras = ", ".join(p for p in [load_info] if p)
            was_loaded = "already loaded" if model_name in before_names else "loaded by check"
            typer.echo(f"  {label}: {model_name} - OK, {was_loaded}{(' (' + extras + ')') if extras else ''}")
        else:
            typer.echo(f"  {label}: {model_name} - FAILED ({result['error']})")

    # Show currently loaded models with VRAM usage
    both_loaded = embed_result["success"] and chat_result["success"]
    configured_loaded = set()
    if running and running.models:
        for model in running.models:
            if model.model in {embed_model, chat_model}:
                configured_loaded.add(model.model)
    expected_configured_count = len({embed_model, chat_model})

    if running and running.models:
        typer.echo("")
        typer.echo("Currently loaded models:")
        for m in running.models:
            size_vram = f"{m.size_vram.real / 1024 / 1024:.0f} MB" if m.size_vram else "N/A"
            size_total = f"{m.size.real / 1024 / 1024:.0f} MB" if m.size else "N/A"
            typer.echo(f"  {m.model}  VRAM: {size_vram}  Total: {size_total}")
    elif running:
        typer.echo("")
        typer.echo("No models currently loaded in GPU memory.")

    # Summary and actionable advice
    typer.echo("")
    if both_loaded and len(configured_loaded) >= expected_configured_count:
        typer.echo(typer.style("OK: configured model residency looks stable after the check.", fg=typer.colors.GREEN))
    elif both_loaded:
        typer.echo(typer.style("WARNING: configured models loaded individually, but not all are resident after the check.", fg=typer.colors.YELLOW))
        typer.echo("")
        typer.echo("  Options to reduce model churn:")
        typer.echo("    - Reduce MAX_CONTEXT_LENGTH (currently {}) to shrink the chat KV cache".format(num_ctx))
        typer.echo("    - Configure OLLAMA_KV_CACHE_TYPE=q8_0 or q4_0 on the Ollama host")
        typer.echo("    - Configure OLLAMA_FLASH_ATTENTION=true on the Ollama host")
        typer.echo("    - Set OLLAMA_EMBED_NUM_GPU=0 to run embedding requests on CPU")
        typer.echo("    - Use a smaller chat model (currently {})".format(chat_model))
        typer.echo("    - Use a smaller embedding model (currently {})".format(embed_model))
    elif not embed_result["success"] or not chat_result["success"]:
        typer.echo(typer.style("FAILED: one or both models failed to load.", fg=typer.colors.RED))
        failed = []
        if not embed_result["success"]:
            failed.append(embed_model)
        if not chat_result["success"]:
            failed.append(chat_model)
        typer.echo(f"  Failed: {', '.join(failed)}")
        typer.echo("")
        typer.echo("  Options:")
        typer.echo("    - Reduce MAX_CONTEXT_LENGTH (currently {})".format(num_ctx))
        typer.echo("    - Configure OLLAMA_KV_CACHE_TYPE=q8_0 or q4_0 on the Ollama host")
        typer.echo("    - Set OLLAMA_EMBED_NUM_GPU=0 to run embedding requests on CPU")
        typer.echo("    - Use smaller models")

    typer.echo("")
    typer.echo("No models were unloaded by this check.")


def _models_action_loaded(model_settings, loop) -> None:
    """Show currently loaded (running) Ollama models with VRAM and processor info."""
    from app.ollama.client import OllamaClientWrapper
    from app.ollama.runtime import running_model_summary, model_name
    from app.core.logging import logger

    async def _run():
        client = OllamaClientWrapper(model_settings, logger)
        try:
            return await running_model_summary(client)
        finally:
            await client.close()

    try:
        running = loop.run_until_complete(_run())
    except Exception as exc:
        typer.echo(f"Failed to list running models: {type(exc).__name__}: {exc}", err=True)
        return

    if not running.get("available"):
        error = running.get("error", {})
        msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        typer.echo(f"Could not inspect running models: {msg}", err=True)
        return

    models = running.get("models", [])
    if not models:
        typer.echo("No models are currently loaded in GPU memory.")
        return

    typer.echo("Currently loaded models:")
    typer.echo(f"  {'Model':<40} {'VRAM':>10} {'Total':>10} {'Processor':<20} {'Context':>10}")
    typer.echo(f"  {'---':<40} {'---':>10} {'---':>10} {'---':<20} {'---':>10}")
    for m in models:
        name = m.get("model") or m.get("name") or "unknown"
        size_vram = m.get("size_vram")
        size_total = m.get("size")
        vram_str = f"{size_vram / 1024 / 1024:.0f} MB" if size_vram else "N/A"
        total_str = f"{size_total / 1024 / 1024:.0f} MB" if size_total else "N/A"
        processor = m.get("processor", "unknown")
        ctx = m.get("context_length")
        ctx_str = str(ctx) if ctx else "N/A"
        typer.echo(f"  {name:<40} {vram_str:>10} {total_str:>10} {processor:<20} {ctx_str:>10}")

    # Show configured model residency
    chat_model = model_settings.ollama_chat_model
    embed_model = model_settings.ollama_embed_model
    loaded_names = {model_name(m) for m in models if isinstance(m, dict)}
    chat_loaded = chat_model in loaded_names
    embed_loaded = embed_model in loaded_names
    typer.echo("")
    typer.echo(f"  Chat model ({chat_model}):      {'resident' if chat_loaded else 'not resident'}")
    typer.echo(f"  Embedding model ({embed_model}): {'resident' if embed_loaded else 'not resident'}")


def _models_action_config(model_settings, env_values: dict[str, str], loop) -> None:
    """Show and edit Ollama runtime configuration."""

    # Try API first for live runtime values
    api_config, api_error = _api_get_ollama_config()
    api_reachable = api_config is not None

    if api_reachable and api_config:
        runtime = api_config.get("runtime", {})
        chat_model = runtime.get("chat_model", model_settings.ollama_chat_model)
        embed_model = runtime.get("embedding_model", model_settings.ollama_embed_model)
        max_ctx = runtime.get("max_context_length", model_settings.max_context_length)
        embed_batch = runtime.get("embedding_batch_size", model_settings.embedding_batch_size)
        embed_num_gpu = runtime.get("embedding_num_gpu", model_settings.ollama_embed_num_gpu)
        input_budget = runtime.get("input_token_budget", int(max_ctx * 0.75))
        host_info = api_config.get("host", {})
        is_local = host_info.get("is_local", False)
        flash_attn = host_info.get("flash_attention")
        kv_cache = host_info.get("kv_cache_type")
    else:
        chat_model = model_settings.ollama_chat_model
        embed_model = model_settings.ollama_embed_model
        max_ctx = model_settings.max_context_length
        embed_batch = model_settings.embedding_batch_size
        embed_num_gpu = model_settings.ollama_embed_num_gpu
        input_budget = int(max_ctx * 0.75)
        is_local = _is_local_host(env_values.get("OLLAMA_HOSTNAME"))
        flash_attn = model_settings.ollama_flash_attention
        kv_cache = model_settings.ollama_kv_cache_type

    embed_gpu_label = "auto (all GPU)" if embed_num_gpu == -1 else ("CPU only" if embed_num_gpu == 0 else f"{embed_num_gpu} layers")

    typer.echo("Current Ollama runtime configuration:")
    typer.echo(f"  Chat model:            {chat_model}")
    typer.echo(f"  Embedding model:       {embed_model}")
    typer.echo(f"  Max context length:     {max_ctx}")
    typer.echo(f"  Input token budget:    {input_budget} (75% of context)")
    typer.echo(f"  Embedding batch size:  {embed_batch}")
    typer.echo(f"  Embedding GPU:         {embed_num_gpu} ({embed_gpu_label})")
    if is_local:
        typer.echo(f"  Flash attention:       {flash_attn}  (server-side)")
        typer.echo(f"  KV cache type:         {kv_cache}  (server-side)")
    else:
        typer.echo(f"  Host mode:             Remote")
    if not api_reachable:
        typer.echo("")
        typer.echo(typer.style("  Note: API is not reachable. Showing .env values; runtime values may differ.", fg=typer.colors.YELLOW))

    typer.echo("")
    typer.echo("Edit which value?")
    typer.echo("  [1] Chat model")
    typer.echo("  [2] Embedding model")
    typer.echo("  [3] Max context length")
    typer.echo("  [4] Embedding batch size")
    typer.echo("  [5] Embedding GPU layers")
    typer.echo("  [a] Apply all changes to running API (hot-reload)")
    typer.echo("  [q] Back to model list")

    choice = typer.prompt("Choose", default="q").strip().lower()

    if choice in ("q", ""):
        return

    patch: dict[str, object] = {}

    if choice == "1":
        new_val = typer.prompt("Chat model name", default=chat_model).strip()
        if new_val and new_val != chat_model:
            patch["chat_model"] = new_val
            _write_env_value(ENV_FILE, "OLLAMA_CHAT_MODEL", new_val)
            env_values["OLLAMA_CHAT_MODEL"] = new_val
            os.environ["OLLAMA_CHAT_MODEL"] = new_val
            typer.echo(f"  Written OLLAMA_CHAT_MODEL={new_val} to .env")
    elif choice == "2":
        new_val = typer.prompt("Embedding model name", default=embed_model).strip()
        if new_val and new_val != embed_model:
            patch["embedding_model"] = new_val
            _write_env_value(ENV_FILE, "OLLAMA_EMBED_MODEL", new_val)
            env_values["OLLAMA_EMBED_MODEL"] = new_val
            os.environ["OLLAMA_EMBED_MODEL"] = new_val
            typer.echo(f"  Written OLLAMA_EMBED_MODEL={new_val} to .env")
    elif choice == "3":
        new_val = typer.prompt("Max context length", type=int, default=max_ctx)
        if new_val != max_ctx:
            patch["max_context_length"] = new_val
            _write_env_value(ENV_FILE, "MAX_CONTEXT_LENGTH", str(new_val))
            env_values["MAX_CONTEXT_LENGTH"] = str(new_val)
            os.environ["MAX_CONTEXT_LENGTH"] = str(new_val)
            typer.echo(f"  Written MAX_CONTEXT_LENGTH={new_val} to .env")
    elif choice == "4":
        new_val = typer.prompt("Embedding batch size", type=int, default=embed_batch)
        if new_val != embed_batch:
            patch["embedding_batch_size"] = new_val
            _write_env_value(ENV_FILE, "EMBEDDING_BATCH_SIZE", str(new_val))
            env_values["EMBEDDING_BATCH_SIZE"] = str(new_val)
            os.environ["EMBEDDING_BATCH_SIZE"] = str(new_val)
            typer.echo(f"  Written EMBEDDING_BATCH_SIZE={new_val} to .env")
    elif choice == "5":
        typer.echo("  Embedding GPU layers: -1 = auto (all GPU), 0 = CPU only, N = N layers on GPU")
        new_val = typer.prompt("Embedding GPU layers", type=int, default=embed_num_gpu)
        if new_val != embed_num_gpu:
            patch["embedding_num_gpu"] = new_val
            _write_env_value(ENV_FILE, "OLLAMA_EMBED_NUM_GPU", str(new_val))
            env_values["OLLAMA_EMBED_NUM_GPU"] = str(new_val)
            os.environ["OLLAMA_EMBED_NUM_GPU"] = str(new_val)
            typer.echo(f"  Written OLLAMA_EMBED_NUM_GPU={new_val} to .env")
    elif choice == "a":
        # Apply all pending .env changes to the running API
        if not api_reachable:
            typer.echo("API is not reachable. Cannot apply changes to running API.", err=True)
            return
        # Build patch from current .env values
        patch["chat_model"] = env_values.get("OLLAMA_CHAT_MODEL", chat_model)
        patch["embedding_model"] = env_values.get("OLLAMA_EMBED_MODEL", embed_model)
        patch["max_context_length"] = int(env_values.get("MAX_CONTEXT_LENGTH", str(max_ctx)))
        patch["embedding_batch_size"] = int(env_values.get("EMBEDDING_BATCH_SIZE", str(embed_batch)))
        patch["embedding_num_gpu"] = int(env_values.get("OLLAMA_EMBED_NUM_GPU", str(embed_num_gpu)))
    else:
        typer.echo(f"Unknown choice: {choice}")
        return

    # Apply patch to running API if there are changes and API is reachable
    if patch and api_reachable:
        typer.echo("Applying runtime changes to running API ...")
        updated, err = _api_patch_runtime(patch)
        if err:
            typer.echo(f"  Failed to apply to API: {err}", err=True)
            typer.echo("  Changes are saved in .env and will take effect on API restart.")
        else:
            typer.echo(typer.style("  Applied to running API (hot-reload).", fg=typer.colors.GREEN))
            typer.echo("  Note: Runtime changes reset when the API restarts. .env values persist.")
    elif patch and not api_reachable:
        typer.echo("API is not reachable. Changes saved to .env will take effect on next API start.")


def _models_action_test(model_settings, loop) -> None:
    """Run the embedding/chat switch test to check GPU residency under load."""
    from app.ollama.client import OllamaClientWrapper
    from app.ollama.runtime import run_performance_test

    # Try API first
    api_reachable = _api_is_reachable()

    chat_model = model_settings.ollama_chat_model
    embed_model = model_settings.ollama_embed_model
    max_ctx = model_settings.max_context_length
    embed_num_gpu = model_settings.ollama_embed_num_gpu

    typer.echo("Switch test - loads embed → chat → embed → chat to detect reload pressure.")
    typer.echo(f"  Chat model:       {chat_model}")
    typer.echo(f"  Embedding model:  {embed_model}")
    typer.echo(f"  Max context:       {max_ctx}")
    typer.echo(f"  Embedding GPU:     {embed_num_gpu}")
    typer.echo("")

    if api_reachable:
        typer.echo("API is reachable. Running switch test via API ...")
        result, err = _api_run_performance_test({
            "chat_model": chat_model,
            "embedding_model": embed_model,
            "max_context_length": max_ctx,
            "embedding_num_gpu": embed_num_gpu,
        })
        if err:
            typer.echo(f"API switch test failed: {err}", err=True)
            typer.echo("Falling back to direct Ollama client ...")
            api_reachable = False
        else:
            # Display API results
            runtime_info = result.get("runtime", {})
            typer.echo(f"  Chat model:       {runtime_info.get('chat_model', chat_model)}")
            typer.echo(f"  Embedding model:  {runtime_info.get('embedding_model', embed_model)}")
            typer.echo(f"  Max context:      {runtime_info.get('max_context_length', max_ctx)}")
            typer.echo(f"  Embedding GPU:    {runtime_info.get('embedding_num_gpu', embed_num_gpu)}")
            typer.echo("")

            initial = result.get("initial", {})
            if initial.get("available") and initial.get("models"):
                typer.echo("Models loaded before test:")
                for m in initial.get("models", []):
                    name = m.get("model") or m.get("name") or "unknown"
                    vram = m.get("size_vram")
                    total = m.get("size")
                    vram_str = f"{vram / 1024 / 1024:.0f} MB" if vram else "N/A"
                    total_str = f"{total / 1024 / 1024:.0f} MB" if total else "N/A"
                    proc = m.get("processor", "unknown")
                    typer.echo(f"  {name}  VRAM: {vram_str}  Total: {total_str}  Processor: {proc}")
            typer.echo("")

            steps = result.get("steps", [])
            for step in steps:
                label = step.get("label", step.get("key", ""))
                model = step.get("model", "")
                success = step.get("success", False)
                load_ms = step.get("load_duration_ms")
                error_msg = step.get("error")
                snapshot = step.get("snapshot", {})
                loaded_count = len(snapshot.get("models", [])) if snapshot.get("available") else "?"

                status_str = typer.style("OK", fg=typer.colors.GREEN) if success else typer.style("FAILED", fg=typer.colors.RED)
                load_str = f"{load_ms:.0f} ms" if load_ms is not None else "N/A"
                typer.echo(f"  {label} ({model}): {status_str}  load: {load_str}  models after: {loaded_count}")
                if error_msg:
                    typer.echo(f"    Error: {error_msg}")

            diagnostics = result.get("diagnostics", {})
            diag_status = diagnostics.get("status", "unknown")
            diag_summary = diagnostics.get("summary", "")
            recommendations = diagnostics.get("recommendations", [])

            typer.echo("")
            color = typer.colors.GREEN if diag_status == "ok" else (typer.colors.YELLOW if diag_status in ("partial", "chat-pressure") else typer.colors.RED)
            typer.echo(f"Diagnostics: {typer.style(diag_status, fg=color)}")
            typer.echo(f"  {diag_summary}")
            if recommendations:
                typer.echo("  Recommendations:")
                for rec in recommendations:
                    typer.echo(f"    - {rec}")

            both_resident = diagnostics.get("both_resident_after_final")
            if both_resident is not None:
                typer.echo(f"  Both models resident after final step: {typer.style(str(both_resident), fg=typer.colors.GREEN if both_resident else typer.colors.RED)}")
            return

    # Fallback: direct Ollama client
    if not api_reachable:
        typer.echo("API is not reachable. Running switch test directly via Ollama ...")

        async def _run_direct():
            from app.core.logging import logger as _logger
            client = OllamaClientWrapper(model_settings, _logger)
            try:
                return await run_performance_test(
                    model_settings,
                    client,
                    chat_model=chat_model,
                    embedding_model=embed_model,
                    max_context_length=max_ctx,
                    embedding_num_gpu=embed_num_gpu,
                )
            finally:
                await client.close()

        try:
            result = loop.run_until_complete(_run_direct())
        except Exception as exc:
            typer.echo(f"Switch test failed: {type(exc).__name__}: {exc}", err=True)
            return

        runtime_info = result.get("runtime", {})
        typer.echo(f"  Chat model:       {runtime_info.get('chat_model', chat_model)}")
        typer.echo(f"  Embedding model:  {runtime_info.get('embedding_model', embed_model)}")
        typer.echo(f"  Max context:      {runtime_info.get('max_context_length', max_ctx)}")
        typer.echo(f"  Embedding GPU:    {runtime_info.get('embedding_num_gpu', embed_num_gpu)}")
        typer.echo("")

        initial = result.get("initial", {})
        if initial.get("available") and initial.get("models"):
            typer.echo("Models loaded before test:")
            for m in initial.get("models", []):
                name = m.get("model") or m.get("name") or "unknown"
                vram = m.get("size_vram")
                total = m.get("size")
                vram_str = f"{vram / 1024 / 1024:.0f} MB" if vram else "N/A"
                total_str = f"{total / 1024 / 1024:.0f} MB" if total else "N/A"
                proc = m.get("processor", "unknown")
                typer.echo(f"  {name}  VRAM: {vram_str}  Total: {total_str}  Processor: {proc}")
        typer.echo("")

        steps = result.get("steps", [])
        for step in steps:
            label = step.get("label", step.get("key", ""))
            model = step.get("model", "")
            success = step.get("success", False)
            load_ms = step.get("load_duration_ms")
            error_msg = step.get("error")
            snapshot = step.get("snapshot", {})
            loaded_count = len(snapshot.get("models", [])) if snapshot.get("available") else "?"

            status_str = typer.style("OK", fg=typer.colors.GREEN) if success else typer.style("FAILED", fg=typer.colors.RED)
            load_str = f"{load_ms:.0f} ms" if load_ms is not None else "N/A"
            typer.echo(f"  {label} ({model}): {status_str}  load: {load_str}  models after: {loaded_count}")
            if error_msg:
                typer.echo(f"    Error: {error_msg}")

        diagnostics = result.get("diagnostics", {})
        diag_status = diagnostics.get("status", "unknown")
        diag_summary = diagnostics.get("summary", "")
        recommendations = diagnostics.get("recommendations", [])

        typer.echo("")
        color = typer.colors.GREEN if diag_status == "ok" else (typer.colors.YELLOW if diag_status in ("partial", "chat-pressure") else typer.colors.RED)
        typer.echo(f"Diagnostics: {typer.style(diag_status, fg=color)}")
        typer.echo(f"  {diag_summary}")
        if recommendations:
            typer.echo("  Recommendations:")
            for rec in recommendations:
                typer.echo(f"    - {rec}")

        both_resident = diagnostics.get("both_resident_after_final")
        if both_resident is not None:
            typer.echo(f"  Both models resident after final step: {typer.style(str(both_resident), fg=typer.colors.GREEN if both_resident else typer.colors.RED)}")


def _models_action_diagnostics(model_settings, loop) -> None:
    """Show Ollama diagnostics: loaded models, residency, and context budget."""
    from app.ollama.client import OllamaClientWrapper
    from app.ollama.runtime import running_model_summary, diagnose_ollama_runtime

    # Try API first
    api_config, api_error = _api_get_ollama_config()
    api_reachable = api_config is not None

    if api_reachable and api_config:
        runtime = api_config.get("runtime", {})
        running_data = api_config.get("running", {})
        diagnostics = api_config.get("diagnostics", {})
        host_info = api_config.get("host", {})
        warnings = api_config.get("warnings", [])

        typer.echo("Ollama diagnostics (from running API):")
        typer.echo("")
        typer.echo(f"  Host:              {host_info.get('base_url', 'unknown')}")
        typer.echo(f"  Host mode:         {'Local' if host_info.get('is_local') else 'Remote'}")
        if host_info.get("is_local"):
            typer.echo(f"  Flash attention:   {host_info.get('flash_attention', 'unknown')}")
            typer.echo(f"  KV cache type:     {host_info.get('kv_cache_type', 'unknown')}")
        typer.echo("")
        typer.echo("  Runtime configuration:")
        typer.echo(f"    Chat model:            {runtime.get('chat_model', 'unknown')}")
        typer.echo(f"    Embedding model:      {runtime.get('embedding_model', 'unknown')}")
        typer.echo(f"    Max context length:    {runtime.get('max_context_length', 'unknown')}")
        typer.echo(f"    Input token budget:   {runtime.get('input_token_budget', 'unknown')} (75% of context)")
        typer.echo(f"    Embedding batch size: {runtime.get('embedding_batch_size', 'unknown')}")
        typer.echo(f"    Embedding GPU:        {runtime.get('embedding_gpu_label', 'unknown')} ({runtime.get('embedding_num_gpu', 'unknown')})")
        typer.echo(f"    Resets on restart:    {runtime.get('resets_on_api_restart', True)}")
        typer.echo("")

        # Loaded models
        if running_data.get("available"):
            models = running_data.get("models", [])
            if models:
                typer.echo("  Loaded models:")
                for m in models:
                    name = m.get("model") or m.get("name") or "unknown"
                    vram = m.get("size_vram")
                    total = m.get("size")
                    vram_str = f"{vram / 1024 / 1024:.0f} MB" if vram else "N/A"
                    total_str = f"{total / 1024 / 1024:.0f} MB" if total else "N/A"
                    proc = m.get("processor", "unknown")
                    ctx = m.get("context_length")
                    ctx_str = str(ctx) if ctx else "N/A"
                    typer.echo(f"    {name}  VRAM: {vram_str}  Total: {total_str}  Processor: {proc}  Context: {ctx_str}")
            else:
                typer.echo("  No models currently loaded.")
        else:
            err = running_data.get("error", {})
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            typer.echo(f"  Could not inspect loaded models: {msg}")

        typer.echo("")
        typer.echo(f"  Chat model loaded:      {running_data.get('chat_model_loaded', '?')}")
        typer.echo(f"  Embedding model loaded: {running_data.get('embedding_model_loaded', '?')}")
        typer.echo(f"  Both models loaded:     {running_data.get('both_configured_models_loaded', '?')}")

        # Diagnostics
        diag_status = diagnostics.get("status", "unknown")
        diag_summary = diagnostics.get("summary", "")
        recommendations = diagnostics.get("recommendations", [])
        color = typer.colors.GREEN if diag_status == "ok" else (typer.colors.YELLOW if diag_status in ("partial", "chat-pressure") else typer.colors.RED)
        typer.echo("")
        typer.echo(f"  Diagnostics: {typer.style(diag_status, fg=color)}")
        typer.echo(f"    {diag_summary}")
        if recommendations:
            typer.echo("    Recommendations:")
            for rec in recommendations:
                typer.echo(f"      - {rec}")

        if warnings:
            typer.echo("")
            typer.echo("  Warnings:")
            for w in warnings:
                typer.echo(f"    - {w}")
        return

    # Fallback: direct Ollama client
    typer.echo("API is not reachable. Querying Ollama directly ...")

    async def _run():
        from app.core.logging import logger as _logger
        client = OllamaClientWrapper(model_settings, _logger)
        try:
            running = await running_model_summary(client)
            return running
        finally:
            await client.close()

    try:
        running = loop.run_until_complete(_run())
    except Exception as exc:
        typer.echo(f"Failed to query Ollama: {type(exc).__name__}: {exc}", err=True)
        return

    chat_model = model_settings.ollama_chat_model
    embed_model = model_settings.ollama_embed_model

    typer.echo("")
    typer.echo("Ollama diagnostics (direct query):")
    typer.echo(f"  Chat model:       {chat_model}")
    typer.echo(f"  Embedding model:  {embed_model}")
    typer.echo(f"  Max context:      {model_settings.max_context_length}")
    typer.echo(f"  Embedding GPU:   {model_settings.ollama_embed_num_gpu}")
    typer.echo("")

    if running.get("available"):
        models = running.get("models", [])
        if models:
            typer.echo("  Loaded models:")
            for m in models:
                name = m.get("model") or m.get("name") or "unknown"
                vram = m.get("size_vram")
                total = m.get("size")
                vram_str = f"{vram / 1024 / 1024:.0f} MB" if vram else "N/A"
                total_str = f"{total / 1024 / 1024:.0f} MB" if total else "N/A"
                proc = m.get("processor", "unknown")
                typer.echo(f"    {name}  VRAM: {vram_str}  Total: {total_str}  Processor: {proc}")
        else:
            typer.echo("  No models currently loaded.")
    else:
        err = running.get("error", {})
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        typer.echo(f"  Could not inspect loaded models: {msg}")

    diagnostics = diagnose_ollama_runtime(chat_model, embed_model, running)
    diag_status = diagnostics.get("status", "unknown")
    diag_summary = diagnostics.get("summary", "")
    recommendations = diagnostics.get("recommendations", [])
    color = typer.colors.GREEN if diag_status == "ok" else (typer.colors.YELLOW if diag_status in ("partial", "chat-pressure") else typer.colors.RED)
    typer.echo("")
    typer.echo(f"  Diagnostics: {typer.style(diag_status, fg=color)}")
    typer.echo(f"    {diag_summary}")
    if recommendations:
        typer.echo("    Recommendations:")
        for rec in recommendations:
            typer.echo(f"      - {rec}")


def _models_action_server(env_values: dict[str, str]) -> None:
    """Show and configure server-side Ollama settings (flash attention, KV cache type)."""
    local_ollama = _is_local_host(env_values.get("OLLAMA_HOSTNAME"))
    if not local_ollama:
        typer.echo("Server-side configuration is only available when Ollama is local/Docker.")
        typer.echo("Change these settings on the Ollama host directly.")
        return

    current_flash = env_values.get("OLLAMA_FLASH_ATTENTION", "1")
    current_kv = env_values.get("OLLAMA_KV_CACHE_TYPE", "q8_0")

    typer.echo("Ollama server-side configuration (requires container restart):")
    typer.echo(f"  OLLAMA_FLASH_ATTENTION = {current_flash}  (enables Flash Attention, reduces memory at larger contexts)")
    typer.echo(f"  OLLAMA_KV_CACHE_TYPE    = {current_kv}  (KV cache quantization: f16=default, q8_0=half memory, q4_0=quarter memory)")
    typer.echo("")
    typer.echo("These are Docker container environment variables. Changes require restarting the Ollama container.")
    typer.echo("")
    typer.echo("  [1] Change OLLAMA_FLASH_ATTENTION")
    typer.echo("  [2] Change OLLAMA_KV_CACHE_TYPE")
    typer.echo("  [r] Restart Ollama container to apply changes")
    typer.echo("  [q] Back to model list")

    choice = typer.prompt("Choose", default="q").strip().lower()

    if choice in ("q", ""):
        return

    if choice == "1":
        new_val = typer.prompt("OLLAMA_FLASH_ATTENTION (0 or 1)", default=current_flash).strip()
        if new_val not in ("0", "1"):
            typer.echo("Value must be 0 or 1.", err=True)
            return
        _write_env_value(ENV_FILE, "OLLAMA_FLASH_ATTENTION", new_val)
        env_values["OLLAMA_FLASH_ATTENTION"] = new_val
        os.environ["OLLAMA_FLASH_ATTENTION"] = new_val
        typer.echo(f"  Set OLLAMA_FLASH_ATTENTION={new_val} in .env")
        typer.echo("  Restart the Ollama container to apply: use [r] here or 'simone host --ollama'")

    elif choice == "2":
        typer.echo("  Options: f16 (default, highest quality), q8_0 (half memory), q4_0 (quarter memory, lowest quality)")
        new_val = typer.prompt("OLLAMA_KV_CACHE_TYPE", default=current_kv).strip()
        if new_val not in ("f16", "q8_0", "q4_0"):
            typer.echo("Warning: Unrecognized KV cache type. Common values are f16, q8_0, q4_0.", err=True)
        _write_env_value(ENV_FILE, "OLLAMA_KV_CACHE_TYPE", new_val)
        env_values["OLLAMA_KV_CACHE_TYPE"] = new_val
        os.environ["OLLAMA_KV_CACHE_TYPE"] = new_val
        typer.echo(f"  Set OLLAMA_KV_CACHE_TYPE={new_val} in .env")
        typer.echo("  Restart the Ollama container to apply: use [r] here or 'simone host --ollama'")

    elif choice == "r":
        if not _docker_available():
            typer.echo("Docker is not available. Cannot restart Ollama container.", err=True)
            return
        if not typer.confirm("Restart the Ollama container? This will unload all models.", default=False):
            typer.echo("Cancelled.")
            return
        typer.echo("Restarting Ollama container ...")
        _run(_build_compose_cmd(ENV_FILE, [COMPOSE_PROD], action="restart", services=["ollama"]), cwd=REPO_ROOT, check=False)
        typer.echo("Ollama container restarted. Models will need to be reloaded.")
    else:
        typer.echo(f"Unknown choice: {choice}")


def _bootstrap_vocabs(foreground: bool, start_services: bool) -> None:
    _ensure_no_bootstrap_vocab_job()

    os.environ["APP_ENV"] = "development"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    local_neo4j, local_ollama = _local_service_flags(env_values)

    if start_services:
        services = [service for service, enabled in (("neo4j", local_neo4j), ("ollama", local_ollama)) if enabled]
        if services:
            if not _docker_available():
                typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
                raise typer.Exit(1)
            typer.echo(f"Starting {' + '.join(services)} containers ...")
            _run(_build_compose_cmd(ENV_FILE, [COMPOSE_PROD], action="up", services=services, build=False), cwd=REPO_ROOT)
            if local_neo4j:
                if not _wait_for_url("http://127.0.0.1:7474", timeout=120, label="Neo4j", verbose=False):
                    raise typer.Exit(1)
            if local_ollama:
                ollama_port = env_values.get("OLLAMA_PORT", "11433")
                if not _wait_for_url(f"http://127.0.0.1:{ollama_port}", timeout=120, label="Ollama", verbose=False):
                    raise typer.Exit(1)
        else:
            typer.echo("Using external Neo4j/Ollama services from .env.")

    process_env = os.environ.copy()
    process_env.update(env_values)
    process_env["APP_ENV"] = "development"
    cmd = [sys.executable, "-m", "app.bootstrap", "initial-vocabs"]

    if foreground:
        _write_bootstrap_vocab_lock(os.getpid())
        try:
            result = _run(cmd, cwd=BACKEND_DIR, check=False, capture_output=False, env=process_env)
            if result.returncode != 0:
                raise typer.Exit(result.returncode)
            return
        finally:
            _remove_bootstrap_vocab_lock(os.getpid())

    BOOTSTRAP_VOCABS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BOOTSTRAP_VOCABS_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Starting initial vocabulary bootstrap\n")
        popen_kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            popen_kwargs["startupinfo"] = startupinfo
        process = subprocess.Popen(
            cmd,
            cwd=str(BACKEND_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=process_env,
            **popen_kwargs,
        )

    _write_bootstrap_vocab_lock(process.pid)
    typer.echo(f"Started initial vocabulary bootstrap in the background (PID {process.pid}).")
    typer.echo(f"Log file: {BOOTSTRAP_VOCABS_LOG.relative_to(REPO_ROOT)}")


@app.command()
def down() -> None:
    """Stop all SIMONE services (containers and local processes)."""
    os.environ.setdefault("APP_ENV", "production")

    typer.echo("Stopping vocabulary bootstrap process ...")
    bootstrap_pids = _stop_bootstrap_vocab_job()
    if bootstrap_pids:
        typer.echo(f"Stopped vocabulary bootstrap processes: {bootstrap_pids}")
    else:
        typer.echo("No vocabulary bootstrap process found.")

    if not _docker_available():
        typer.echo("Warning: Docker not available. Skipping container shutdown.", err=True)
    else:
        stacks = [
            ("SIMONE", ENV_FILE, [COMPOSE_PROD]),
        ]

        stopped_any_stack = False
        for label, env_file, compose_files in stacks:
            if not env_file.exists() or not all(compose_file.exists() for compose_file in compose_files):
                continue
            try:
                if not _compose_has_containers(env_file, compose_files):
                    continue
                typer.echo(f"Stopping {label} containers ...")
                cmd = _build_compose_cmd(env_file, compose_files, action="down")
                _run(cmd, cwd=REPO_ROOT, check=False)
                stopped_any_stack = True
            except Exception:
                pass
        if not stopped_any_stack:
            typer.echo("No SIMONE Docker containers found.")

    # Kill local dev processes
    typer.echo("Stopping local API process ...")
    api_pids = sorted(set(_find_pids_by_cmdline("uvicorn app.main:fastapi_app") + _find_pids_by_window_title("SIMONE API")))
    if api_pids:
        _kill_pids(api_pids)
        typer.echo(f"Stopped API processes: {api_pids}")
    else:
        _kill_windows_by_title("SIMONE API")
        typer.echo("No local API process found.")

    typer.echo("Stopping local frontend process ...")
    frontend_pids = _find_pids_by_cmdline("npm run dev")
    if not frontend_pids:
        frontend_pids = _find_pids_by_cmdline("vite")
    frontend_pids = sorted(set(frontend_pids + _find_pids_by_window_title("SIMONE Frontend")))
    if frontend_pids:
        _kill_pids(frontend_pids)
        typer.echo(f"Stopped frontend processes: {frontend_pids}")
    else:
        _kill_windows_by_title("SIMONE Frontend")
        typer.echo("No local frontend process found.")

    typer.echo("SIMONE is down.")


@app.command("signin-ollama")
def signin_ollama() -> None:
    """Run Ollama Cloud sign-in inside the Docker Ollama container."""
    if not _docker_available():
        typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
        raise typer.Exit(1)
    os.environ["APP_ENV"] = "production"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    if not _is_local_host(env_values.get("OLLAMA_HOSTNAME")):
        typer.echo("Ollama is configured as an external service in .env.")
        typer.echo("Run 'ollama signin' on the machine that hosts Ollama.")
        return
    _compose_exec(ENV_FILE, [COMPOSE_PROD], "ollama", ["ollama", "signin"])


@app.command("backup-neo4j")
def backup_neo4j() -> None:
    """Create a Cypher backup of the development Neo4j database."""
    if not _docker_available():
        typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
        raise typer.Exit(1)
    os.environ["APP_ENV"] = "development"
    _backup_neo4j()


@app.command("reset-neo4j")
def reset_neo4j(
    backup: bool = typer.Option(False, "--backup", help="Create a Cypher backup before resetting"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Reset without an interactive confirmation prompt"),
) -> None:
    """Reset the local Neo4j data directory used by development and production."""
    if not _docker_available():
        typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
        raise typer.Exit(1)
    os.environ["APP_ENV"] = "development"

    typer.echo("WARNING: This will delete the local Neo4j database in data/docker/neo4j/data.")
    should_backup = backup
    if not backup and not yes:
        should_backup = typer.confirm("Create a backup before resetting?", default=True)

    if not yes and not typer.confirm("Reset local Neo4j data?", default=False):
        typer.echo("Reset cancelled.")
        return

    if should_backup:
        _backup_neo4j()

    typer.echo("Stopping Neo4j containers ...")
    for env_file, compose_files in (
        (ENV_FILE, [COMPOSE_PROD]),
    ):
        if env_file.exists() and all(compose_file.exists() for compose_file in compose_files):
            _compose_stop_services(env_file, compose_files, ["neo4j"])

    typer.echo("Removing Neo4j data contents ...")
    _reset_neo4j_data_dir()
    typer.echo("Neo4j database reset complete.")
    typer.echo("The next startup will initialize Neo4j with the current .env credentials.")


@app.command("restore-neo4j")
def restore_neo4j(
    backup_file: Path = typer.Argument(..., help="Path to a .cypher backup file"),
) -> None:
    """Restore a Cypher backup into the development Neo4j database."""
    if not _docker_available():
        typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
        raise typer.Exit(1)
    if not backup_file.exists():
        typer.echo(f"Backup file not found: {backup_file}", err=True)
        raise typer.Exit(1)

    os.environ["APP_ENV"] = "development"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    user = env_values.get("NEO4J_USER", "neo4j")
    password = env_values.get("NEO4J_PASSWORD", "")

    typer.echo("Ensuring Neo4j container is running ...")
    _run(_build_compose_cmd(ENV_FILE, [COMPOSE_PROD], action="up", services=["neo4j"], build=False), cwd=REPO_ROOT, check=False)

    container_id = _compose_container_id(ENV_FILE, [COMPOSE_PROD], "neo4j")
    if not container_id:
        typer.echo("No running development Neo4j container was found.", err=True)
        raise typer.Exit(1)

    typer.echo("Copying backup into Neo4j container ...")
    copy_result = _run(["docker", "cp", str(backup_file), f"{container_id}:/tmp/restore.cypher"], cwd=REPO_ROOT, check=False)
    if copy_result.returncode != 0:
        typer.echo("Failed to copy backup file into Neo4j container.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Restoring Neo4j graph from {backup_file} ...")
    result = _compose_exec(
        ENV_FILE,
        [COMPOSE_PROD],
        "neo4j",
        ["cypher-shell", "-u", user, "-p", password, "-f", "/tmp/restore.cypher"],
        capture_output=True,
    )
    if result.returncode != 0:
        typer.echo("Restore failed. Check .env Neo4j credentials and backup contents.", err=True)
        if result.stderr:
            typer.echo(result.stderr.strip(), err=True)
        raise typer.Exit(1)
    typer.echo("Restore completed successfully.")


@app.command()
def status() -> None:
    """Show the current status of SIMONE services."""
    os.environ.setdefault("APP_ENV", "production")
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    if not _docker_available():
        typer.echo("Docker is not available.")
    else:
        typer.echo("Docker containers:")
        result = _run(
            _compose_base_cmd(ENV_FILE, [COMPOSE_PROD]) + ["ps", "--format", "table {{.Service}}\t{{.Status}}\t{{.Ports}}"],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.stdout:
            typer.echo(result.stdout)
        else:
            typer.echo("  (no containers running)")

    typer.echo("")
    typer.echo("Local dev processes (used by 'simone dev', not by production 'simone up'):")
    api_pids = _find_pids_by_cmdline("uvicorn app.main:fastapi_app")
    frontend_pids = _find_pids_by_cmdline("npm run dev") or _find_pids_by_cmdline("vite")

    typer.echo(f"  API (uvicorn):     {'running (PID ' + str(api_pids) + ')' if api_pids else 'not running'}")
    typer.echo(f"  Frontend (vite):   {'running (PID ' + str(frontend_pids) + ')' if frontend_pids else 'not running'}")

    typer.echo("")
    typer.echo("Background jobs:")
    _print_vocab_bootstrap_status()

    _print_api_health()

    typer.echo("")
    typer.echo("Service URLs:")
    typer.echo(f"  Frontend:    {_frontend_url()}")
    typer.echo(f"  API Docs:    {API_URL}")
    typer.echo(f"  Neo4j:       {_neo4j_browser_url()}")


if __name__ == "__main__":
    app()
