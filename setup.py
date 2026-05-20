#!/usr/bin/env python3
"""
SIMONE Setup Assistant

Run this script after cloning the repository. It checks prerequisites,
guides you through installing anything that's missing, and then sets up
the project automatically.

Usage:
    python setup.py
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    OK = "\033[92m" if sys.platform != "win32" else ""
    WARN = "\033[93m" if sys.platform != "win32" else ""
    FAIL = "\033[91m" if sys.platform != "win32" else ""
    INFO = "\033[94m" if sys.platform != "win32" else ""
    BOLD = "\033[1m" if sys.platform != "win32" else ""
    END = "\033[0m" if sys.platform != "win32" else ""


def _print(status: str, message: str) -> None:
    color = {"OK": Colors.OK, "WARN": Colors.WARN, "FAIL": Colors.FAIL, "INFO": Colors.INFO, "BOLD": Colors.BOLD}.get(status, "")
    symbol = {"OK": "✓", "WARN": "⚠", "FAIL": "✗", "INFO": "ℹ", "BOLD": "→"}.get(status, " ")
    print(f"{color}[{symbol}] {message}{Colors.END}")


def _input(prompt_text: str) -> str:
    try:
        return input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        print("")
        sys.exit(0)


def _get_os() -> str:
    system = platform.system()
    if system == "Windows":
        return "windows"
    if system == "Darwin":
        return "macos"
    return "linux"


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _wait_for_enter() -> None:
    _input("\nPress Enter to continue...")


def step_check_docker(os_name: str) -> bool:
    """Step 1: Check Docker."""
    print("")
    print("─" * 60)
    _print("BOLD", "Step 1 / 3 — Docker Desktop")
    print("─" * 60)

    if _command_exists("docker"):
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                _print("OK", "Docker and Docker Compose are already installed.")
                return True
        except Exception:
            pass

    _print("FAIL", "Docker is not installed or Docker Compose plugin is missing.")
    print("")
    print("Please install Docker Desktop:")
    print(f"  {Colors.INFO}https://www.docker.com/get-started/{Colors.END}")
    print("")
    if os_name == "windows":
        print("  1. Download the installer from the link above.")
        print("  2. Run the installer and follow the setup wizard.")
        print("  3. Restart your computer if prompted.")
    elif os_name == "macos":
        print("  1. Download the installer from the link above.")
        print("  2. Open the .dmg and drag Docker to Applications.")
        print("  3. Launch Docker Desktop from Applications.")
    else:
        print("  1. Follow the official guide at the link above.")
        print("  2. Or install via your package manager, e.g.:")
        print("     sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin")
    print("")
    _wait_for_enter()
    return step_check_docker(os_name)


def step_check_uv(os_name: str) -> bool:
    """Step 2: Check uv."""
    print("")
    print("─" * 60)
    _print("BOLD", "Step 2 / 3 — uv (Python package manager)")
    print("─" * 60)

    if _command_exists("uv"):
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip().splitlines()[0]
                _print("OK", f"uv is already installed ({version}).")
                return True
        except Exception:
            pass

    _print("FAIL", "uv is not installed or not on PATH.")
    print("")
    print("Please install uv:")
    print(f"  {Colors.INFO}https://docs.astral.sh/uv/getting-started/installation/{Colors.END}")
    print("")
    if os_name == "windows":
        print("  1. Open PowerShell and run:")
        print('     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"')
        print("  2. Restart your terminal.")
    else:
        print("  1. Open a terminal and run:")
        print('     curl -LsSf https://astral.sh/uv/install.sh | sh')
        print("  2. Restart your terminal or run: source $HOME/.local/bin/env")
    print("")
    _wait_for_enter()
    return step_check_uv(os_name)


def step_check_node(os_name: str) -> bool:
    """Step 3: Check Node.js (optional)."""
    print("")
    print("─" * 60)
    _print("BOLD", "Step 3 / 3 — Node.js + npm (optional, only needed for development mode)")
    print("─" * 60)

    node_ok = False
    npm_ok = False

    if _command_exists("node"):
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                _print("OK", f"Node.js is installed ({result.stdout.strip()}).")
                node_ok = True
        except Exception:
            pass
    else:
        _print("FAIL", "Node.js is not installed.")

    if _command_exists("npm"):
        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                _print("OK", f"npm is installed ({result.stdout.strip()}).")
                npm_ok = True
        except Exception:
            pass
    else:
        _print("FAIL", "npm is not installed.")

    if node_ok and npm_ok:
        return True

    print("")
    print("Node.js is only required if you plan to use `simone dev` (development mode).")
    print("You can skip this step if you only need `simone up` (production mode).")
    print("")
    print("To install Node.js:")
    print(f"  {Colors.INFO}https://nodejs.org/{Colors.END}")
    print("")
    if os_name == "windows":
        print("  1. Download the LTS installer from the link above.")
        print("  2. Run the .msi installer and follow the wizard.")
        print("  3. Restart your terminal.")
    elif os_name == "macos":
        print("  1. Download the LTS installer from the link above.")
        print("  2. Run the .pkg installer and follow the wizard.")
        print("  Or use Homebrew: brew install node")
    else:
        print("  1. Use your package manager, e.g.:")
        print("     sudo apt-get install nodejs npm")
        print("  2. Or download from https://nodejs.org/")
    print("")
    choice = _input("Have you installed Node.js? (y/n/skip): ").strip().lower()
    if choice in ("y", "yes"):
        return step_check_node(os_name)
    return False


def run_uv_sync() -> bool:
    """Run uv sync in the backend directory."""
    backend_dir = Path(__file__).resolve().parent / "backend"
    if not backend_dir.exists():
        _print("FAIL", "backend/ directory not found. Are you running this from the repo root?")
        return False

    print("")
    print("─" * 60)
    _print("BOLD", "Installing Python dependencies...")
    print("─" * 60)
    print(f"Running: uv sync (in {backend_dir})")
    print("")

    try:
        result = subprocess.run(
            ["uv", "sync"],
            cwd=backend_dir,
            check=False,
        )
        if result.returncode != 0:
            _print("FAIL", "uv sync failed. Please check the error messages above.")
            return False
        _print("OK", "Python dependencies installed successfully.")
        return True
    except FileNotFoundError:
        _print("FAIL", "Could not run 'uv sync'. Is uv on your PATH?")
        return False


def print_next_steps(has_node: bool) -> None:
    """Print the final next-steps message."""
    print("")
    print("=" * 60)
    _print("OK", "Setup complete! SIMONE is ready to run.")
    print("=" * 60)
    print("")
    print("Start the application:")
    if sys.platform == "win32":
        print(f"  {Colors.BOLD}simone.bat up{Colors.END}       # Production mode (all services in Docker)")
        if has_node:
            print(f"  {Colors.BOLD}simone.bat dev{Colors.END}      # Development mode (local API + frontend with hot reload)")
        print(f"  {Colors.BOLD}simone.bat up --gpu{Colors.END}   # Production mode with GPU support")
        print("")
        print("Other useful commands:")
        print(f"  {Colors.BOLD}simone.bat down{Colors.END}     # Stop all services")
        print(f"  {Colors.BOLD}simone.bat status{Colors.END}   # Check what's running")
    else:
        print(f"  {Colors.BOLD}./simone up{Colors.END}       # Production mode (all services in Docker)")
        if has_node:
            print(f"  {Colors.BOLD}./simone dev{Colors.END}      # Development mode (local API + frontend with hot reload)")
        print(f"  {Colors.BOLD}./simone up --gpu{Colors.END}   # Production mode with GPU support")
        print("")
        print("Other useful commands:")
        print(f"  {Colors.BOLD}./simone down{Colors.END}     # Stop all services")
        print(f"  {Colors.BOLD}./simone status{Colors.END}   # Check what's running")
    print("")
    print("Service URLs once started:")
    print("  Frontend:    http://127.0.0.1:3000")
    print("  API Docs:    http://127.0.0.1:8000/docs")
    print("  Neo4j:       http://127.0.0.1:7474/browser/")
    print("")


def main() -> int:
    os_name = _get_os()
    os_label = {"windows": "Windows", "macos": "macOS", "linux": "Linux"}.get(os_name, os_name)

    print("")
    print("=" * 60)
    print("  SIMONE Setup Assistant")
    print(f"  Detected OS: {os_label}")
    print("=" * 60)
    print("")
    print("This script will guide you through installing the prerequisites")
    print("and setting up the project. Just follow the prompts.")
    print("")

    # Step 1: Docker
    if not step_check_docker(os_name):
        return 1

    # Step 2: uv
    if not step_check_uv(os_name):
        return 1

    # Step 3: Node.js (optional)
    has_node = step_check_node(os_name)

    # Step 4: uv sync
    if not run_uv_sync():
        return 1

    # Final message
    print_next_steps(has_node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
