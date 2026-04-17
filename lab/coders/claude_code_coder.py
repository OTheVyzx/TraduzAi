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
import tempfile
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

        # Escreve prompt em arquivo temp para evitar limite de args no Windows
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as prompt_file:
            prompt_file.write(full_prompt)
            prompt_path = Path(prompt_file.name)

        try:
            result = subprocess.run(
                [
                    claude_path,
                    "-p",
                    full_prompt,
                    "--output-format",
                    "json",
                    "--model",
                    self.model,
                ],
                capture_output=True,
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
        finally:
            prompt_path.unlink(missing_ok=True)

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[-500:]
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
                error=f"claude CLI retornou exit {result.returncode}: {stderr_snippet}",
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
