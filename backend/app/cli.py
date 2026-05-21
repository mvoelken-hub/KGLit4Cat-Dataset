import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
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

ENV_PROD = REPO_ROOT / ".env.production"
ENV_PROD_EXAMPLE = REPO_ROOT / ".env.production.example"
ENV_DEV = REPO_ROOT / ".env.development"
ENV_DEV_EXAMPLE = REPO_ROOT / ".env.development.example"

COMPOSE_PROD = REPO_ROOT / "docker-compose.yml"
COMPOSE_DEV = REPO_ROOT / "docker-compose.dev.yml"
COMPOSE_GPU = REPO_ROOT / "docker-compose.gpu.yml"

API_URL = "http://127.0.0.1:8000/docs"
HEALTH_URL = "http://127.0.0.1:8000/api/v1/health"
FRONTEND_URL = "http://127.0.0.1:3000"
NEO4J_BROWSER_URL = "http://127.0.0.1:7474/browser/"


def _run(
    cmd: list[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True)


def _check_command(name: str) -> bool:
    """Check if a command is available on PATH."""
    result = subprocess.run(["where", name] if sys.platform == "win32" else ["which", name], capture_output=True)
    return result.returncode == 0


def _docker_available() -> bool:
    return _check_command("docker")


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


def _compose_exec(env_file: Path, compose_files: list[Path], service: str, command: list[str]) -> subprocess.CompletedProcess:
    cmd = _compose_base_cmd(env_file, compose_files) + ["exec", "-T", service, *command]
    return _run(cmd, cwd=REPO_ROOT, check=False, capture_output=False)


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


def _ensure_ollama_signin(env_file: Path, compose_files: list[Path]) -> None:
    status_code, payload, error = _get_json_url(HEALTH_URL, timeout=30)
    if error:
        typer.echo(f"Could not check Ollama Cloud sign-in status yet: {error}", err=True)
        return
    if not _is_ollama_chat_unauthorized(payload):
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

    status_code, payload, error = _get_json_url(HEALTH_URL, timeout=30)
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
    status_code, payload, error = _get_json_url(HEALTH_URL, timeout=30)
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
) -> bool:
    typer.echo(f"Waiting for {label} at {url} ...")
    start = time.time()
    attempt = 0
    last_error = ""
    last_details_at = -10
    previous_logs = ""
    if verbose and env_file and compose_files:
        previous_logs = _compose_logs(env_file, compose_files, ["api"])
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
        if verbose and env_file and compose_files:
            current_logs = _compose_logs(env_file, compose_files, ["api"])
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
    typer.echo("")
    typer.echo("SIMONE is starting up. You can access the services at:")
    typer.echo(f"  Frontend:    {FRONTEND_URL}")
    typer.echo(f"  API Docs:    {API_URL}")
    typer.echo(f"  Neo4j:       {NEO4J_BROWSER_URL}")
    typer.echo("")


def _compose_files_for_mode(dev: bool, gpu: bool) -> list[Path]:
    files = [COMPOSE_DEV if dev else COMPOSE_PROD]
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
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
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
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU support for Ollama"),
    build: bool = typer.Option(False, "--build", help="Build images before starting containers"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show startup progress, container status, and API logs while waiting"),
) -> None:
    """Start SIMONE in production mode (all services in Docker)."""
    if not _docker_available():
        typer.echo("Error: Docker is not installed or not on PATH.", err=True)
        raise typer.Exit(1)

    _ensure_env_file(ENV_PROD, ENV_PROD_EXAMPLE)

    compose_files = _compose_files_for_mode(dev=False, gpu=gpu)
    cmd = _build_compose_cmd(ENV_PROD, compose_files, action="up", build=build)

    typer.echo(f"Starting production stack ({'GPU' if gpu else 'CPU'}) ...")
    typer.echo(" ".join(cmd))
    _run(cmd, cwd=REPO_ROOT, capture_output=not verbose)

    api_ready = _wait_for_url(API_URL, timeout=120, label="API", verbose=verbose, env_file=ENV_PROD, compose_files=compose_files)
    if api_ready:
        _ensure_ollama_signin(ENV_PROD, compose_files)
    _print_links()


@app.command()
def dev(
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU support for Ollama"),
) -> None:
    """Start SIMONE in development mode (Neo4j + Ollama in Docker; API + frontend locally)."""
    if not _docker_available():
        typer.echo("Error: Docker is not installed or not on PATH.", err=True)
        raise typer.Exit(1)

    _ensure_env_file(ENV_DEV, ENV_DEV_EXAMPLE)

    # Dependency sync
    if not _check_command("uv"):
        typer.echo("Error: 'uv' is not installed or not on PATH. Install it from https://docs.astral.sh/uv/", err=True)
        raise typer.Exit(1)

    if not (BACKEND_DIR / ".venv").exists():
        typer.echo("Running uv sync in backend/ ...")
        _run(["uv", "sync"], cwd=BACKEND_DIR)
    else:
        typer.echo("Backend virtual environment already exists. Skipping uv sync.")

    if not _check_command("npm"):
        typer.echo("Error: 'npm' is not installed or not on PATH. Install Node.js from https://nodejs.org/", err=True)
        raise typer.Exit(1)

    if not (FRONTEND_DIR / "node_modules").exists():
        typer.echo("Running npm install in frontend/ ...")
        _run(["npm", "install"], cwd=FRONTEND_DIR)
    else:
        typer.echo("Frontend node_modules already exists. Skipping npm install.")

    # Start infrastructure containers
    compose_files = _compose_files_for_mode(dev=True, gpu=gpu)
    cmd = _build_compose_cmd(ENV_DEV, compose_files, action="up", services=["neo4j", "ollama"], build=True)

    typer.echo(f"Starting infrastructure containers ({'GPU' if gpu else 'CPU'}) ...")
    typer.echo(" ".join(cmd))
    _run(cmd, cwd=REPO_ROOT)

    # Start local API in a new visible terminal window
    api_cmd = (
        'uv run --env-file ../.env.development uvicorn app.main:fastapi_app '
        '--host 127.0.0.1 --port 8000 --reload'
    )
    typer.echo("Starting local API with hot reload ...")
    if sys.platform == "win32":
        subprocess.Popen(
            ["cmd", "/c", "start", "SIMONE API", "powershell", "-NoExit", "-Command", api_cmd],
            cwd=str(BACKEND_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        subprocess.Popen(
            api_cmd,
            cwd=str(BACKEND_DIR),
            shell=True,
        )

    # Start local frontend in a new visible terminal window
    frontend_cmd = "npm run dev -- --host 127.0.0.1"
    typer.echo("Starting local frontend dev server ...")
    if sys.platform == "win32":
        subprocess.Popen(
            ["cmd", "/c", "start", "SIMONE Frontend", "powershell", "-NoExit", "-Command", frontend_cmd],
            cwd=str(FRONTEND_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        subprocess.Popen(
            frontend_cmd,
            cwd=str(FRONTEND_DIR),
            shell=True,
        )

    _wait_for_url(API_URL, timeout=120, label="API", verbose=True, env_file=ENV_DEV, compose_files=compose_files)
    _wait_for_url(FRONTEND_URL, timeout=120, label="Frontend", verbose=True, env_file=ENV_DEV, compose_files=compose_files)
    _print_links()


@app.command()
def down() -> None:
    """Stop all SIMONE services (containers and local processes)."""
    if not _docker_available():
        typer.echo("Warning: Docker not available. Skipping container shutdown.", err=True)
    else:
        stacks = [
            ("production", ENV_PROD, [COMPOSE_PROD]),
            ("development", ENV_DEV, [COMPOSE_DEV]),
        ]
        if COMPOSE_GPU.exists():
            stacks.extend(
                [
                    ("production GPU", ENV_PROD, [COMPOSE_PROD, COMPOSE_GPU]),
                    ("development GPU", ENV_DEV, [COMPOSE_DEV, COMPOSE_GPU]),
                ]
            )

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
    api_pids = _find_pids_by_cmdline("uvicorn app.main:fastapi_app")
    if api_pids:
        _kill_pids(api_pids)
        typer.echo(f"Stopped API processes: {api_pids}")
    else:
        typer.echo("No local API process found.")

    typer.echo("Stopping local frontend process ...")
    frontend_pids = _find_pids_by_cmdline("npm run dev")
    if not frontend_pids:
        frontend_pids = _find_pids_by_cmdline("vite")
    if frontend_pids:
        _kill_pids(frontend_pids)
        typer.echo(f"Stopped frontend processes: {frontend_pids}")
    else:
        typer.echo("No local frontend process found.")

    typer.echo("SIMONE is down.")


@app.command("signin-ollama")
def signin_ollama() -> None:
    """Run Ollama Cloud sign-in inside the Docker Ollama container."""
    if not _docker_available():
        typer.echo("Error: Docker is not installed or not on PATH.", err=True)
        raise typer.Exit(1)
    _ensure_env_file(ENV_PROD, ENV_PROD_EXAMPLE)
    _compose_exec(ENV_PROD, [COMPOSE_PROD], "ollama", ["ollama", "signin"])


@app.command()
def status() -> None:
    """Show the current status of SIMONE services."""
    if not _docker_available():
        typer.echo("Docker is not available.")
        return

    typer.echo("Docker containers:")
    result = _run(
        _compose_base_cmd(ENV_PROD, [COMPOSE_PROD]) + ["ps", "--format", "table {{.Service}}\t{{.Status}}\t{{.Ports}}"],
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
    typer.echo(f"  Frontend:    {FRONTEND_URL}")
    typer.echo(f"  API Docs:    {API_URL}")
    typer.echo(f"  Neo4j:       {NEO4J_BROWSER_URL}")


if __name__ == "__main__":
    app()
