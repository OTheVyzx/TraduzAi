"""Utilitários base para agentes Claude do Lab.

- Roteamento target_file → .claude/agents/*.md
- Criação do cliente Anthropic
- Verificação de disponibilidade (API key + pacote)
"""
from __future__ import annotations

import os
from pathlib import Path

# Mapeamento prefixo de arquivo → agent markdown
# Avaliado em ordem; o primeiro match ganha.
AGENT_FILE_ROUTING: list[tuple[str, str]] = [
    ("pipeline/typesetter/", "typesetter-expert.md"),
    ("pipeline/layout/", "typesetter-expert.md"),
    ("pipeline/translator/", "translator-expert.md"),
    ("pipeline/vision_stack/", "vision-stack-expert.md"),
    ("pipeline/ocr/", "vision-stack-expert.md"),
    ("pipeline/inpainter/", "vision-stack-expert.md"),
    ("lab/critics/", "typesetter-expert.md"),  # critics vivem no domínio do pipeline
]

# reviewer_id → dica de domínio para routing secundário
REVIEWER_DOMAIN_HINTS: dict[str, str] = {
    "python_senior_reviewer": "pipeline/",
    "rust_senior_reviewer": "src-tauri/",
    "react_ts_senior_reviewer": "src/",
    "tauri_boundary_reviewer": "src-tauri/",
    "integration_architect": "",
}

_repo_root_cache: Path | None = None


def find_repo_root() -> Path:
    """Localiza a raiz do repositório (contém .claude/)."""
    global _repo_root_cache
    if _repo_root_cache is not None:
        return _repo_root_cache

    # lab/agents/base.py → lab/ → repo_root
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / ".claude").exists():
        _repo_root_cache = candidate
        return _repo_root_cache

    # Walk up até 6 níveis como fallback
    p = Path(__file__).resolve()
    for _ in range(6):
        p = p.parent
        if (p / ".claude").exists():
            _repo_root_cache = p
            return _repo_root_cache

    _repo_root_cache = candidate
    return _repo_root_cache


def get_agent_prompt(
    target_file: str = "",
    touched_domains: list[str] | None = None,
) -> str | None:
    """Carrega o system prompt do agente correto para o arquivo/domínio dado.

    Prioridade:
    1. Prefixo de `target_file` (mais específico)
    2. Prefixo de cada entrada em `touched_domains`
    Retorna None se nenhum agente for encontrado.
    """
    agents_dir = find_repo_root() / ".claude" / "agents"
    if not agents_dir.exists():
        return None

    # 1. Por target_file
    for prefix, agent_file in AGENT_FILE_ROUTING:
        if target_file.startswith(prefix):
            path = agents_dir / agent_file
            if path.exists():
                return path.read_text(encoding="utf-8")

    # 2. Por touched_domains
    for domain in (touched_domains or []):
        for prefix, agent_file in AGENT_FILE_ROUTING:
            if prefix.rstrip("/") in domain:
                path = agents_dir / agent_file
                if path.exists():
                    return path.read_text(encoding="utf-8")

    return None


def get_api_key() -> str:
    """Lê ANTHROPIC_API_KEY do ambiente."""
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def is_claude_available() -> bool:
    """Verdadeiro apenas se API key presente E pacote anthropic instalado."""
    if not get_api_key():
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def make_client():
    """Cria e retorna um cliente Anthropic. Lança RuntimeError se não disponível."""
    import anthropic  # pode lançar ImportError

    key = get_api_key()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada. "
            "Defina a variável de ambiente antes de iniciar o TraduzAi."
        )
    return anthropic.Anthropic(api_key=key)


def read_file_slice(path: Path, max_lines: int = 300) -> str:
    """Lê as primeiras `max_lines` linhas de um arquivo (seguro para contexto Claude)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[:max_lines])
    except Exception:
        return ""


__all__ = [
    "AGENT_FILE_ROUTING",
    "REVIEWER_DOMAIN_HINTS",
    "find_repo_root",
    "get_agent_prompt",
    "get_api_key",
    "is_claude_available",
    "make_client",
    "read_file_slice",
]
