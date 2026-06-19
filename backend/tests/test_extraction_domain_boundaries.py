import ast
from pathlib import Path


DOMAIN_DIR = Path(__file__).resolve().parents[1] / "app" / "domain" / "extraction"
FORBIDDEN_IMPORTS = (
    "app.services",
    "app.repositories",
    "app.ollama",
)


def test_extraction_domain_stays_runtime_free():
    offenders: list[str] = []
    for path in DOMAIN_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(FORBIDDEN_IMPORTS):
                    offenders.append(f"{path.name}: from {module}")
                if module == "app.core.task_registry":
                    imported = {alias.name for alias in node.names}
                    if "TaskRegistry" in imported:
                        offenders.append(f"{path.name}: from {module} import TaskRegistry")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_IMPORTS):
                        offenders.append(f"{path.name}: import {alias.name}")
    assert offenders == []
