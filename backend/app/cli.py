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

API_URL = "http://127.0.0.1:8000/docs"
HEALTH_URL = "http://127.0.0.1:8000/api/v1/health"
NEO4J_BROWSER_URL = "http://127.0.0.1:7474/browser/"
NEO4J_DATA_DIR = REPO_ROOT / "data" / "docker" / "neo4j" / "data"
NEO4J_BACKUP_DIR = REPO_ROOT / ".backups" / "neo4j"


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


def _is_local_host(hostname: str | None) -> bool:
    return not hostname or hostname.strip().lower() in {"localhost", "127.0.0.1", "::1"}


def _local_service_flags(env_values: dict[str, str]) -> tuple[bool, bool]:
    return _is_local_host(env_values.get("NEO4J_HOSTNAME")), _is_local_host(env_values.get("OLLAMA_HOSTNAME"))


def _frontend_url(env_values: Optional[dict[str, str]] = None) -> str:
    values = env_values if env_values is not None else _read_env_file(ENV_FILE)
    return f"http://127.0.0.1:{values.get('FRONTEND_PORT', '3000')}"


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
    typer.echo("")
    typer.echo("SIMONE is starting up. You can access the services at:")
    typer.echo(f"  Frontend:    {frontend_url}")
    typer.echo(f"  API Docs:    {API_URL}")
    typer.echo(f"  Neo4j:       {NEO4J_BROWSER_URL}")
    typer.echo("")


def _compose_files_for_mode() -> list[Path]:
    return [COMPOSE_PROD]


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
) -> None:
    """Start SIMONE in production mode (API/frontend in Docker)."""
    if not _docker_available():
        typer.echo("Error: Docker is not running. Please start Docker Desktop.", err=True)
        raise typer.Exit(1)

    os.environ["APP_ENV"] = "production"
    _ensure_env_file(ENV_FILE, ENV_EXAMPLE)
    env_values = _read_env_file(ENV_FILE)
    local_neo4j, local_ollama = _local_service_flags(env_values)

    compose_files = _compose_files_for_mode()
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
    compose_files = _compose_files_for_mode()
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
    # Start local API in a new visible terminal window
    api_cmd = (
        "$env:APP_ENV='development'; uv run --env-file ../.env uvicorn app.main:fastapi_app "
        '--host 127.0.0.1 --port 8000 --reload'
    )
    existing_api = _find_pids_by_cmdline("uvicorn app.main:fastapi_app") or _find_pids_by_window_title("SIMONE API")
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

    if not use_docker_frontend:
        # Start local frontend in a new visible terminal window
        existing_frontend = _find_pids_by_cmdline("npm run dev") or _find_pids_by_cmdline("vite") or _find_pids_by_window_title("SIMONE Frontend")
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

    compose_files = _compose_files_for_mode()
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


@app.command()
def down() -> None:
    """Stop all SIMONE services (containers and local processes)."""
    os.environ.setdefault("APP_ENV", "production")
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
        return

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

    _print_api_health()

    typer.echo("")
    typer.echo("Service URLs:")
    typer.echo(f"  Frontend:    {_frontend_url()}")
    typer.echo(f"  API Docs:    {API_URL}")
    typer.echo(f"  Neo4j:       {NEO4J_BROWSER_URL}")


if __name__ == "__main__":
    app()
