import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="SIMONE CLI — Manage the SIMONE application stack")

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
FRONTEND_URL = "http://127.0.0.1:3000"
NEO4J_BROWSER_URL = "http://127.0.0.1:7474/browser/"


def _run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


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


def _wait_for_url(url: str, timeout: int = 120, label: str = "service") -> bool:
    typer.echo(f"Waiting for {label} at {url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    typer.echo(f"{label} is ready.")
                    return True
        except Exception:
            pass
        time.sleep(2)
    typer.echo(f"{label} was not reachable after {timeout} seconds.", err=True)
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
                f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
        )
        pids = [int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip().isdigit()]
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

@app.command()
def up(
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU support for Ollama"),
) -> None:
    """Start SIMONE in production mode (all services in Docker)."""
    if not _docker_available():
        typer.echo("Error: Docker is not installed or not on PATH.", err=True)
        raise typer.Exit(1)

    _ensure_env_file(ENV_PROD, ENV_PROD_EXAMPLE)

    compose_files = _compose_files_for_mode(dev=False, gpu=gpu)
    cmd = _build_compose_cmd(ENV_PROD, compose_files, action="up", build=True)

    typer.echo(f"Starting production stack ({'GPU' if gpu else 'CPU'}) ...")
    typer.echo(" ".join(cmd))
    _run(cmd, cwd=REPO_ROOT)

    _wait_for_url(API_URL, timeout=120, label="API")
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

    _wait_for_url(API_URL, timeout=120, label="API")
    _wait_for_url(FRONTEND_URL, timeout=120, label="Frontend")
    _print_links()


@app.command()
def down() -> None:
    """Stop all SIMONE services (containers and local processes)."""
    if not _docker_available():
        typer.echo("Warning: Docker not available. Skipping container shutdown.", err=True)
    else:
        # Try to stop production containers
        if COMPOSE_PROD.exists():
            try:
                cmd = _build_compose_cmd(ENV_PROD, [COMPOSE_PROD], action="down")
                typer.echo("Stopping production containers ...")
                _run(cmd, cwd=REPO_ROOT, check=False)
            except Exception:
                pass

        # Try to stop dev containers
        if COMPOSE_DEV.exists():
            try:
                cmd = _build_compose_cmd(ENV_DEV, [COMPOSE_DEV], action="down")
                typer.echo("Stopping development containers ...")
                _run(cmd, cwd=REPO_ROOT, check=False)
            except Exception:
                pass

        # Try to stop GPU overlay containers
        if COMPOSE_GPU.exists():
            try:
                for env_file in (ENV_PROD, ENV_DEV):
                    if env_file.exists():
                        cmd = _build_compose_cmd(env_file, [COMPOSE_PROD if env_file == ENV_PROD else COMPOSE_DEV, COMPOSE_GPU], action="down")
                        _run(cmd, cwd=REPO_ROOT, check=False)
            except Exception:
                pass

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


@app.command()
def status() -> None:
    """Show the current status of SIMONE services."""
    if not _docker_available():
        typer.echo("Docker is not available.")
        return

    typer.echo("Docker containers:")
    result = _run(_docker_compose_cmd() + ["ps", "--format", "table {{.Service}}\t{{.Status}}\t{{.Ports}}"], cwd=REPO_ROOT, check=False)
    if result.stdout:
        typer.echo(result.stdout)
    else:
        typer.echo("  (no containers running)")

    typer.echo("")
    typer.echo("Local processes:")
    api_pids = _find_pids_by_cmdline("uvicorn app.main:fastapi_app")
    frontend_pids = _find_pids_by_cmdline("npm run dev") or _find_pids_by_cmdline("vite")

    typer.echo(f"  API (uvicorn):     {'running (PID ' + str(api_pids) + ')' if api_pids else 'not running'}")
    typer.echo(f"  Frontend (vite):   {'running (PID ' + str(frontend_pids) + ')' if frontend_pids else 'not running'}")

    typer.echo("")
    typer.echo("Service URLs:")
    typer.echo(f"  Frontend:    {FRONTEND_URL}")
    typer.echo(f"  API Docs:    {API_URL}")
    typer.echo(f"  Neo4j:       {NEO4J_BROWSER_URL}")


if __name__ == "__main__":
    app()
