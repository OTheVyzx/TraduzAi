"""Runner unificado para invocar o Claude Code CLI (`claude -p`) a partir do Lab.

Centraliza a logica de:
- localizar o binario nativo `claude.exe` (pulando o shim `.cmd` do npm no Windows)
- chamar `subprocess.run` com stdin fechado e encoding UTF-8
- parsear a saida JSON (`--output-format json`)
- devolver `(texto, erro)` em uma interface simples

Usado por reviewer_agent, coder_agent e claude_code_coder. Substitui as chamadas
ao Anthropic SDK (que exigem ANTHROPIC_API_KEY com saldo) — a auth aqui e a
mesma da assinatura Pro/Max via `claude setup-token` / OAuth.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

CLAUDE_CLI_TIMEOUT = 300  # segundos


def find_claude_cli() -> str | None:
    """Retorna o path do CLI `claude` ou None."""
    return shutil.which("claude")


def is_cli_available() -> bool:
    """Verdadeiro se o binario `claude` esta no PATH."""
    return find_claude_cli() is not None


def resolve_claude_invocation(claude_path: str) -> list[str]:
    """Resolve a melhor forma de invocar o Claude Code CLI.

    No Windows, `npm install -g` gera um shim `.cmd` que executa via `cmd.exe`,
    herdando o limite de 8191 chars para linha de comando. Para prompts grandes
    isso quebra (erro "Linha de comando muito longa"). Solucao: localizar o
    binario nativo `claude.exe` empacotado em
    `node_modules/@anthropic-ai/claude-code/bin/claude.exe` e invoca-lo direto
    via `CreateProcess` (limite ~32k chars).
    """
    path = Path(claude_path)
    suffix = path.suffix.lower()
    if suffix not in {".cmd", ".bat"}:
        return [claude_path]

    candidate_dirs: list[Path] = [path.parent]
    try:
        npm_root = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
        )
        if npm_root.returncode == 0 and npm_root.stdout.strip():
            candidate_dirs.append(Path(npm_root.stdout.strip()).parent)
    except (OSError, subprocess.SubprocessError):
        pass

    import os as _os
    for env_key in ("APPDATA", "LOCALAPPDATA", "ProgramFiles"):
        env_val = _os.environ.get(env_key)
        if env_val:
            candidate_dirs.append(Path(env_val) / "npm")
            candidate_dirs.append(Path(env_val) / "nodejs")

    seen: set[Path] = set()
    for base in candidate_dirs:
        try:
            base_resolved = base.resolve()
        except OSError:
            continue
        if base_resolved in seen or not base_resolved.exists():
            continue
        seen.add(base_resolved)
        exe_candidate = (
            base_resolved
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "bin"
            / "claude.exe"
        )
        if exe_candidate.exists():
            return [str(exe_candidate)]

    return [claude_path]


def invoke_claude_cli(
    prompt: str,
    *,
    system: str | None = None,
    model: str = "claude-sonnet-4-5",
    cwd: Path | None = None,
    timeout: int = CLAUDE_CLI_TIMEOUT,
) -> tuple[str, str]:
    """Executa `claude -p` e retorna `(texto_resultado, mensagem_erro)`.

    Em sucesso: `(text, "")`.
    Em falha: `("", "mensagem descrevendo o erro")`.

    O prompt e passado via argv posicional (o CLI interpreta `-p` como flag
    boolean --print e o proximo argv nao-flag como o prompt do usuario).
    O system prompt vai via `--system-prompt` quando fornecido.
    """
    claude_path = find_claude_cli()
    if not claude_path:
        return (
            "",
            "Claude Code CLI (`claude`) nao encontrado no PATH. "
            "Instale via `npm install -g @anthropic-ai/claude-code` e faca "
            "login com `claude setup-token` (assinatura) ou export "
            "ANTHROPIC_API_KEY.",
        )

    invocation = resolve_claude_invocation(claude_path)
    args: list[str] = invocation + [
        "-p",
        prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--permission-mode",
        "bypassPermissions",
    ]
    if system:
        args.extend(["--system-prompt", system])

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired:
        return ("", f"Claude CLI excedeu timeout de {timeout}s.")
    except Exception as exc:
        return ("", f"Erro ao invocar Claude CLI: {exc}")

    stdout = (result.stdout or "").strip()

    # Tenta parsear JSON (--output-format json).
    data: dict | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                data = parsed
        except ValueError:
            # Pode ser ndjson (uma linha por objeto)
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("type") == "result":
                        data = obj
                        break
                except ValueError:
                    continue

    if data is not None:
        if data.get("is_error"):
            api_status = data.get("api_error_status")
            res_field = str(data.get("result", ""))[:400]
            return ("", f"Claude CLI api_error={api_status}: {res_field}")
        text = str(data.get("result", "")).strip()
        if text:
            return (text, "")
        # JSON valido mas sem result — provavelmente erro silencioso
        subtype = data.get("subtype", "?")
        return ("", f"Claude CLI respondeu sem 'result' (subtype={subtype}).")

    if result.returncode != 0:
        stderr = (result.stderr or "")[:500].strip()
        tail = stderr or stdout[:500].strip() or "(sem saida)"
        return ("", f"Claude CLI exit {result.returncode}: {tail}")

    # Saida nao-JSON mas exit 0 — devolve raw.
    return (stdout, "") if stdout else ("", "Claude CLI retornou saida vazia.")


__all__ = [
    "find_claude_cli",
    "is_cli_available",
    "resolve_claude_invocation",
    "invoke_claude_cli",
    "CLAUDE_CLI_TIMEOUT",
]
