"""Layering fitness function (review 2026-10-14, msa#34): `core/` depends only on its own
ports (core/ports.py) and on `gauntlet/`, never on `adapters/`, the REPL or the CLI -- the
composition roots (cli.py, evals/) wire the adapters in. Checked on every import statement,
including ones inside functions, so a lazy import cannot slip past."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = ("adapters", "repl", "cli", "evals")


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


CORE_FILES = sorted((ROOT / "core").glob("*.py"))


@pytest.mark.parametrize("path", CORE_FILES, ids=[p.name for p in CORE_FILES])
def test_core_imports_no_adapter_repl_or_cli(path):
    bad = [m for m in _imported_modules(path) if m.split(".")[0] in FORBIDDEN]
    assert bad == [], f"{path.name} imports {bad}"


def test_the_check_sees_nested_imports(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("def f():\n    from adapters.model import X\n    import repl.session\n")
    assert _imported_modules(sample) == ["adapters.model", "repl.session"]
