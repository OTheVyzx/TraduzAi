import ast
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[1]


def iter_python_files():
    for path in SERVER_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts or path.parts[-2:] == ("tests", "test_storage_contract.py"):
            continue
        yield path


def test_storage_dir_io_is_centralized_in_storage_module():
    offenders = []
    for path in iter_python_files():
        if path.name in {"storage.py", "config.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                offenders.append(f"{path}:{node.lineno}: open()")
            if isinstance(node, ast.Attribute) and node.attr in {"storage_dir"}:
                offenders.append(f"{path}:{node.lineno}: settings.storage_dir")
    assert offenders == []


def test_claim_transition_is_centralized_in_queue_module():
    offenders = []
    for path in iter_python_files():
        if path.name == "queue.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "status=\"claimed\"" in text or "status = \"claimed\"" in text or "status='claimed'" in text:
            offenders.append(str(path))
    assert offenders == []
