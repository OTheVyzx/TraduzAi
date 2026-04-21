"""Claude Code Coder — gera patches via `claude -p` CLI (opt-in, usa creditos API).

O custo aparece SOMENTE quando o usuario clica "Gerar patch" na UI.
Zero chamadas durante a rodada normal do Lab.

Requer `claude` no PATH (Claude Code CLI). Se nao encontrado, retorna erro
imediatamente sem tentar alternativa.

Fluxo:
1. Tenta `build_local_patch_from_hint` deterministico (sem LLM).
2. Se nao resolver, monta prompt e invoca:
       claude -p "<prompt>" --output-format json
3. Extrai bloco diff da saida JSON.
4. Retorna PatchProposal (dry_run=True sempre).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from lab.coders.base import PatchProposal, build_local_patch_from_hint
from lab.coders.ollama_coder import (
    WINDOWS_PIPELINE_CONSTRAINTS,
    _build_user_prompt,
    _estimate_confidence,
    _extract_diff_from_response,
    _extract_rationale,
)

CLAUDE_CLI_TIMEOUT = 300  # segundos — Claude Code nao tem limite de contexto pequeno


def _find_claude_cli() -> str | None:
    """Retorna o path do CLI `claude` ou None se nao encontrado."""
    return shutil.which("claude")


def _resolve_claude_invocation(claude_path: str) -> list[str]:
    """Resolve a melhor forma de invocar o Claude Code CLI.

    No Windows, `npm install -g` gera um shim `.cmd` que executa via `cmd.exe`,
    herdando o limite de 8191 chars para linha de comando. Passar o prompt como
    argv quebra em prompts grandes (erro "Linha de comando muito longa").

    Solucao: se o path apontar para um `.cmd`/`.bat`, procura o binario nativo
    `claude.exe` empacotado em `node_modules/@anthropic-ai/claude-code/bin/` e
    invoca ele direto — isso usa `CreateProcess` (limite ~32k chars) sem passar
    por cmd.exe. Fallback: executar o `.cmd` como antes.
    """
    path = Path(claude_path)
    suffix = path.suffix.lower()
    if suffix not in {".cmd", ".bat"}:
        return [claude_path]

    candidate_dirs: list[Path] = [path.parent]
    # Pergunta ao npm onde fica o root global.
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


def _build_claude_prompt(proposal: dict, file_content: str) -> str:
    """Monta prompt completo para `claude -p`."""
    system_block = (
        "Voce e um engenheiro senior de Python especializado em pipelines de IA para "
        "traducao de manga. Sua tarefa e gerar um patch unified-diff minimo e correto.\n\n"
        + WINDOWS_PIPELINE_CONSTRAINTS
        + "\n\nFORMATO DE RESPOSTA:\n"
        "1. Bloco ```diff ... ``` com o unified diff pronto para `git apply`.\n"
        "2. Paragrafo curto explicando o que mudou e por que.\n"
        "NÃO gere codigo fora do bloco diff. NÃO altere interfaces publicas sem necessidade."
    )
    user_block = _build_user_prompt(proposal, file_content)
    return f"{system_block}\n\n---\n\n{user_block}"


class ClaudeCodeCoder:
    """Coder que usa o Claude Code CLI para gerar patches de alta qualidade."""

    coder_id = "claude_code_coder"

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        timeout: int = CLAUDE_CLI_TIMEOUT,
    ) -> None:
        self.model = model
        self.timeout = timeout

    def propose_patch(self, proposal: dict, repo_root: Path) -> PatchProposal:
        import datetime

        proposal_id = str(proposal.get("proposal_id", "unknown"))
        generated_at = datetime.datetime.utcnow().isoformat() + "Z"
        target_file_rel = str(proposal.get("target_file", "")).strip()

        # Fase 1: patch deterministico sem LLM
        local = build_local_patch_from_hint(proposal, repo_root)
        if local is not None:
            local.generated_at_iso = generated_at
            return local

        # Fase 2: CLI Claude Code
        claude_path = _find_claude_cli()
        if claude_path is None:
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                files_affected=[target_file_rel],
                rationale="",
                author=self.coder_id,
                confidence=0.0,
                model_used=self.model,
                generated_at_iso=generated_at,
                dry_run=True,
                error=(
                    "Claude Code CLI (`claude`) nao encontrado no PATH. "
                    "Instale via `npm install -g @anthropic-ai/claude-code` e autentique."
                ),
            )

        file_content = ""
        if target_file_rel:
            target_path = (repo_root / target_file_rel).resolve()
            if target_path.exists():
                try:
                    file_content = target_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    pass

        full_prompt = _build_claude_prompt(proposal, file_content)

        # Invoca direto via `node cli.js` quando disponivel, para evitar o shim
        # `.cmd` do npm que herda o limite de 8191 chars do cmd.exe no Windows.
        invocation = _resolve_claude_invocation(claude_path)
        cli_args = invocation + [
            "-p",
            full_prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--permission-mode",
            "bypassPermissions",
        ]
        try:
            result = subprocess.run(
                cli_args,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=str(repo_root),
            )
        except subprocess.TimeoutExpired:
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                files_affected=[target_file_rel],
                rationale="",
                author=self.coder_id,
                confidence=0.0,
                model_used=self.model,
                generated_at_iso=generated_at,
                dry_run=True,
                error=f"Claude Code CLI excedeu timeout de {self.timeout}s.",
            )
        except Exception as exc:
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                files_affected=[target_file_rel],
                rationale="",
                author=self.coder_id,
                confidence=0.0,
                model_used=self.model,
                generated_at_iso=generated_at,
                dry_run=True,
                error=f"Erro ao invocar Claude Code CLI: {exc}",
            )

        if result.returncode != 0:
            import os as _os
            stderr_snippet = (result.stderr or "")[:600].strip()
            stdout_snippet = (result.stdout or "")[:600].strip()
            # Se stdout parece JSON, tenta extrair campos relevantes (is_error, error, subtype).
            parsed_hint = ""
            if stdout_snippet.startswith("{"):
                try:
                    data = json.loads(result.stdout)
                    if isinstance(data, dict):
                        is_err = data.get("is_error")
                        err_status = data.get("api_error_status")
                        subtype = data.get("subtype")
                        res_field = data.get("result", "")
                        parsed_hint = (
                            f"is_error={is_err} subtype={subtype} "
                            f"api_status={err_status} result={str(res_field)[:200]}"
                        )
                except (ValueError, TypeError):
                    pass
            detail = parsed_hint or stderr_snippet or stdout_snippet or "(sem saida)"
            invocation_label = Path(invocation[0]).name if invocation else "claude"
            env_probe = (
                f"USERPROFILE={_os.environ.get('USERPROFILE', '?')}, "
                f"APPDATA={_os.environ.get('APPDATA', '?')[:50]}, "
                f"HOME={_os.environ.get('HOME', '?')}"
            )
            detail = f"[via {invocation_label}, prompt={len(full_prompt)} chars, env: {env_probe}] {detail}"
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                files_affected=[target_file_rel],
                rationale="",
                author=self.coder_id,
                confidence=0.0,
                model_used=self.model,
                generated_at_iso=generated_at,
                dry_run=True,
                error=f"claude CLI retornou exit {result.returncode}: {detail}",
            )

        # Tenta parsear saida JSON do Claude Code CLI
        response_text = _parse_cli_output(result.stdout)
        diff = _extract_diff_from_response(response_text)
        rationale = _extract_rationale(response_text)
        confidence = _estimate_confidence(diff, response_text)

        return PatchProposal(
            proposal_id=proposal_id,
            patch_unified_diff=diff,
            files_affected=[target_file_rel] if target_file_rel else [],
            rationale=rationale,
            author=self.coder_id,
            confidence=confidence,
            model_used=self.model,
            generated_at_iso=generated_at,
            dry_run=True,
            error="" if diff else "Claude Code nao gerou um diff valido na resposta.",
        )


def _parse_cli_output(stdout: str) -> str:
    """Extrai texto da saida JSON do `claude --output-format json`.

    O CLI emite um objeto JSON com campo `result` (modo nao-interativo).
    Fallback: usa stdout bruto se nao parsear.
    """
    stdout = stdout.strip()
    if not stdout:
        return ""
    try:
        data = json.loads(stdout)
        # Formato stream-json: ultimo objeto tem type=result
        if isinstance(data, dict):
            return str(data.get("result", data.get("content", stdout)))
        # Formato ndjson: procura ultima linha com type=result
        if isinstance(data, list):
            for item in reversed(data):
                if isinstance(item, dict) and item.get("type") == "result":
                    return str(item.get("result", ""))
    except (json.JSONDecodeError, ValueError):
        # Tenta ndjson (uma linha por objeto)
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("type") == "result":
                    return str(obj.get("result", ""))
            except (json.JSONDecodeError, ValueError):
                continue
    return stdout


__all__ = ["ClaudeCodeCoder"]
