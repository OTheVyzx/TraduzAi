"""Utilitários base para agentes Claude do Lab.

- Roteamento target_file → .claude/agents/*.md
- Seleção de prompt especialista por domínio real
- Verificação de disponibilidade do Claude CLI
"""
from __future__ import annotations

import os
from pathlib import Path

# Mapeamento prefixo de arquivo → agent markdown.
# Avaliado em ordem; o primeiro match ganha.
AGENT_FILE_ROUTING: list[tuple[str, str]] = [
    ("pipeline/typesetter/", "typesetter-expert.md"),
    ("pipeline/layout/", "typesetter-expert.md"),
    ("pipeline/translator/", "translator-expert.md"),
    ("pipeline/vision_stack/runtime.py", "vision-stack-expert.md"),
    ("pipeline/main.py", "vision-stack-expert.md"),
    ("pipeline/vision_stack/detector.py", "detect-expert.md"),
    ("pipeline/vision_stack/ocr.py", "ocr-expert.md"),
    ("pipeline/vision_stack/inpainter.py", "inpaint-expert.md"),
    ("pipeline/vision_stack/", "vision-stack-expert.md"),
    ("pipeline/ocr_legacy/", "ocr-expert.md"),
    ("pipeline/ocr/", "ocr-expert.md"),
    ("pipeline/inpainter_legacy/", "inpaint-expert.md"),
    ("pipeline/inpainter/", "inpaint-expert.md"),
    ("pipeline/", "vision-stack-expert.md"),
    ("lab/critics/typeset_critic.py", "typesetter-expert.md"),
    ("lab/critics/translation_critic.py", "translator-expert.md"),
    ("lab/critics/ocr_critic.py", "ocr-expert.md"),
    ("lab/critics/inpaint_critic.py", "inpaint-expert.md"),
    ("src/lib/tauri.ts", "tauri-frontend-expert.md"),
    ("src/lib/stores/", "tauri-frontend-expert.md"),
    ("src-tauri/src/commands/", "tauri-frontend-expert.md"),
    ("src-tauri/src/lib.rs", "tauri-frontend-expert.md"),
    ("src-tauri/src/main.rs", "tauri-frontend-expert.md"),
    ("src-tauri/src/", "tauri-frontend-expert.md"),
    ("src/", "tauri-frontend-expert.md"),
]

# reviewer_id → dica de domínio para routing secundário
REVIEWER_DOMAIN_HINTS: dict[str, str] = {
    "python_senior_reviewer": "pipeline/",
    "rust_senior_reviewer": "src-tauri/src/",
    "react_ts_senior_reviewer": "src/",
    "tauri_boundary_reviewer": "src-tauri/src/",
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

    normalized_target = _normalize_route_value(target_file)

    # 1. Por target_file
    for prefix, agent_file in AGENT_FILE_ROUTING:
        if _matches_route(normalized_target, prefix):
            path = agents_dir / agent_file
            if path.exists():
                return path.read_text(encoding="utf-8")

    # 2. Por touched_domains
    for domain in (touched_domains or []):
        normalized_domain = _normalize_route_value(domain)
        for prefix, agent_file in AGENT_FILE_ROUTING:
            if _matches_route(normalized_domain, prefix):
                path = agents_dir / agent_file
                if path.exists():
                    return path.read_text(encoding="utf-8")

    return None


def _normalize_route_value(value: str) -> str:
    return (value or "").strip().replace("\\", "/").lstrip("./")


def _matches_route(value: str, prefix: str) -> bool:
    if not value:
        return False

    normalized_prefix = prefix.replace("\\", "/").rstrip("/")
    return (
        value == normalized_prefix
        or value.startswith(prefix)
        or value.startswith(normalized_prefix + "/")
    )


def get_api_key() -> str:
    """Le ANTHROPIC_API_KEY do ambiente (mantido por compatibilidade)."""
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def is_claude_available() -> bool:
    """Verdadeiro se o CLI `claude` esta acessivel no PATH.

    Migrado do Anthropic SDK para o Claude Code CLI: a auth agora vem de
    `claude setup-token` (OAuth da assinatura) ou CLAUDE_CODE_OAUTH_TOKEN /
    ANTHROPIC_API_KEY — gerenciada pelo proprio CLI. Nao dependemos mais do
    pacote `anthropic` em runtime.
    """
    from lab.agents.claude_cli_runner import is_cli_available
    return is_cli_available()


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
    "read_file_slice",
]
