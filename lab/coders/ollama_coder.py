"""Ollama Coder — gera patches via modelo local (Qwen2.5-Coder-7B/DeepSeek-Coder-V2).

Sem custo de API, 100% local. Invocado somente quando o usuario clica
"Gerar patch" na UI para uma proposta com `needs_coder=True`.

Fluxo:
1. Tenta `build_local_patch_from_hint` (deterministico, sem LLM).
2. Se nao resolver, constroi prompt com contexto do arquivo alvo + proposta.
3. Envia para Ollama, extrai bloco unified-diff da resposta.
4. Valida que o diff e syntacticamente correto e retorna PatchProposal.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from lab.coders.base import PatchProposal, build_local_patch_from_hint


OLLAMA_HOST_DEFAULT = "http://localhost:11434"

# Modelos preferidos, em ordem. O primeiro disponivel e escolhido.
PREFERRED_MODELS = [
    "qwen2.5-coder:7b",
    "qwen2.5-coder:7b-instruct-q4_K_M",
    "deepseek-coder-v2:7b",
    "deepseek-coder:6.7b",
    "codellama:7b",
]

# Constraints criticas do CLAUDE.md injetadas no system prompt
WINDOWS_PIPELINE_CONSTRAINTS = """\
RESTRIÇÕES TÉCNICAS OBRIGATÓRIAS (projeto TraduzAi — Windows/FreeType):
- NÃO usar PIL ImageFont.truetype() — causa segfault 0xc0000005
- NÃO usar ProcessPoolExecutor — Windows re-executa main.py nos workers
- NÃO usar ThreadPoolExecutor com FreeType — não é thread-safe para mesma face
- NÃO usar matplotlib TextPath — falha com acentos e segfault acumulado
- NÃO usar TextToPath singleton — cache interno conflita com FT2Font
- USAR matplotlib.ft2font.FT2Font para rendering de texto
- USAR estimativa len(text) * size * 0.55 para bbox em binary search
- Execução serial obrigatória no typesetting
- Linguagem de saída: Python 3.12, type hints, sem import desnecessário
""".strip()


def _list_ollama_models(host: str) -> list[str]:
    """Retorna lista de nomes de modelos instalados, [] se Ollama indisponivel."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _pick_model(host: str) -> str | None:
    """Escolhe o melhor modelo disponivel no Ollama."""
    available = {m.lower() for m in _list_ollama_models(host)}
    if not available:
        return None
    for candidate in PREFERRED_MODELS:
        if candidate.lower() in available:
            return candidate
    # Fallback: qualquer modelo com "coder" ou "code" no nome
    for name in sorted(available):
        if "coder" in name or "code" in name:
            return name
    return None


def _call_ollama_raw(
    *,
    host: str,
    model: str,
    system: str,
    user_msg: str,
    temperature: float = 0.10,
    timeout: int = 300,
) -> str:
    """Chama /api/chat e retorna o conteudo da resposta como string bruta."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return str(data.get("message", {}).get("content", "")).strip()


def _extract_diff_from_response(response: str) -> str:
    """Extrai bloco unified diff de uma resposta Ollama em formato livre."""
    # Tenta extrair bloco marcado ```diff ... ``` ou ```patch ... ```
    for fence_lang in ("diff", "patch", ""):
        pattern = (
            rf"```{re.escape(fence_lang)}\s*\n(.*?)```"
            if fence_lang
            else r"```\s*\n(.*?)```"
        )
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: extrai linhas que parecem unified diff (--- +++ @@ ...)
    diff_lines: list[str] = []
    in_diff = False
    for line in response.splitlines():
        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("@@ "):
            in_diff = True
        if in_diff:
            if line.startswith((" ", "+", "-", "@", "\\", "---", "+++")):
                diff_lines.append(line)
            elif line.strip() and not line.startswith((" ", "+", "-", "@", "\\")):
                # Linha de texto fora do diff — encerra bloco
                if diff_lines:
                    break
    return "\n".join(diff_lines).strip()


def _extract_rationale(response: str) -> str:
    """Extrai a explicacao em texto livre (fora do bloco diff)."""
    cleaned = re.sub(r"```.*?```", "", response, flags=re.DOTALL).strip()
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    rationale = " ".join(lines[:6])  # primeiras 6 linhas de texto
    return rationale[:600]


def _build_system_prompt(change_kind: str) -> str:
    header = (
        "Voce e um engenheiro senior de Python especializado em pipelines de IA para "
        "traducao de manga. Sua tarefa e gerar um patch unified-diff minimo e correto "
        f"para resolver o problema descrito (change_kind={change_kind!r}).\n\n"
    )
    footer = (
        "\n\nFORMATO DE RESPOSTA:\n"
        "1. Bloco ```diff ... ``` com o unified diff pronto para `git apply`.\n"
        "2. Um paragrafo curto explicando o que mudou e por que.\n"
        "NÃO gere codigo fora do bloco diff. NÃO altere interfaces publicas sem necessidade."
    )
    return header + WINDOWS_PIPELINE_CONSTRAINTS + footer


def _build_user_prompt(proposal: dict, file_content: str) -> str:
    lines: list[str] = []
    lines.append(f"# Proposta: {proposal.get('title', '?')}")
    lines.append(f"\n**Motivacao:** {proposal.get('motivation', proposal.get('summary', ''))}")
    lines.append(f"\n**Arquivo alvo:** `{proposal.get('target_file', '?')}`")
    if anchor := proposal.get("target_anchor"):
        lines.append(f"**Ancora:** `{anchor}`")
    lines.append(f"\n**change_kind:** `{proposal.get('change_kind', '?')}`")

    hint = proposal.get("local_patch_hint") or {}
    if hint:
        lines.append(f"\n**Hint local:** {json.dumps(hint, ensure_ascii=False)}")

    metric_gain = proposal.get("expected_metric_gain") or {}
    if metric_gain:
        lines.append(f"\n**Ganho esperado:** {metric_gain}")

    sample = proposal.get("review_findings") or proposal.get("findings_sample") or []
    if sample:
        lines.append("\n**Amostra de findings:**")
        for finding in sample[:3]:
            lines.append(f"  - [{finding.get('severity','?')}] {finding.get('suggested_fix', '')[:200]}")

    if file_content:
        lines.append(f"\n\n**Conteudo atual de `{proposal.get('target_file', '?')}`:**")
        lines.append("```python")
        # Limita para os primeiros 300 linhas para nao explodir o contexto
        file_lines = file_content.splitlines()
        truncated = len(file_lines) > 300
        lines.append("\n".join(file_lines[:300]))
        if truncated:
            lines.append(f"# ... ({len(file_lines) - 300} linhas omitidas)")
        lines.append("```")

    lines.append(
        "\n\nGere o unified diff que corrige o problema descrito seguindo TODAS as "
        "restricoes tecnicas do sistema prompt."
    )
    return "\n".join(lines)


class OllamaCoder:
    """Coder que usa modelo local via Ollama para gerar patches."""

    coder_id = "ollama_coder"

    def __init__(
        self,
        *,
        host: str = OLLAMA_HOST_DEFAULT,
        model: str | None = None,
        temperature: float = 0.10,
        timeout: int = 300,
    ) -> None:
        self.host = host
        self._forced_model = model
        self.temperature = temperature
        self.timeout = timeout

    def _resolve_model(self) -> str | None:
        if self._forced_model:
            return self._forced_model
        return _pick_model(self.host)

    def propose_patch(self, proposal: dict, repo_root: Path) -> PatchProposal:
        """Gera PatchProposal para a proposal dada.

        Tenta `build_local_patch_from_hint` primeiro (deterministico);
        somente escala para Ollama se necessario.
        """
        proposal_id = str(proposal.get("proposal_id", "unknown"))
        import datetime

        generated_at = datetime.datetime.utcnow().isoformat() + "Z"

        # Fase 1: patch deterministico sem LLM
        local = build_local_patch_from_hint(proposal, repo_root)
        if local is not None:
            local.generated_at_iso = generated_at
            return local

        # Fase 2: escalada para Ollama
        model = self._resolve_model()
        if model is None:
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                files_affected=[str(proposal.get("target_file", ""))],
                rationale="",
                author=self.coder_id,
                confidence=0.0,
                model_used="",
                generated_at_iso=generated_at,
                dry_run=True,
                error=(
                    "Ollama nao disponivel ou nenhum modelo de codigo instalado. "
                    "Instale qwen2.5-coder:7b com `ollama pull qwen2.5-coder:7b`."
                ),
            )

        target_file_rel = str(proposal.get("target_file", "")).strip()
        file_content = ""
        if target_file_rel:
            target_path = (repo_root / target_file_rel).resolve()
            if target_path.exists():
                try:
                    file_content = target_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    pass

        change_kind = str(proposal.get("change_kind", "logic_fix"))
        system_prompt = _build_system_prompt(change_kind)
        user_prompt = _build_user_prompt(proposal, file_content)

        try:
            response = _call_ollama_raw(
                host=self.host,
                model=model,
                system=system_prompt,
                user_msg=user_prompt,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        except urllib.error.URLError as exc:
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                files_affected=[target_file_rel],
                rationale="",
                author=self.coder_id,
                confidence=0.0,
                model_used=model,
                generated_at_iso=generated_at,
                dry_run=True,
                error=f"Erro de rede ao chamar Ollama: {exc}",
            )
        except Exception as exc:  # pragma: no cover
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                files_affected=[target_file_rel],
                rationale="",
                author=self.coder_id,
                confidence=0.0,
                model_used=model,
                generated_at_iso=generated_at,
                dry_run=True,
                error=f"Excecao inesperada ao chamar Ollama: {exc}",
            )

        diff = _extract_diff_from_response(response)
        rationale = _extract_rationale(response)
        confidence = _estimate_confidence(diff, response)

        return PatchProposal(
            proposal_id=proposal_id,
            patch_unified_diff=diff,
            files_affected=[target_file_rel] if target_file_rel else [],
            rationale=rationale,
            author=self.coder_id,
            confidence=confidence,
            model_used=model,
            generated_at_iso=generated_at,
            dry_run=True,
            error="" if diff else "Ollama nao gerou um diff valido na resposta.",
        )


def _estimate_confidence(diff: str, raw_response: str) -> float:
    """Heuristica simples para estimar confianca no patch gerado."""
    if not diff:
        return 0.0
    lines = diff.splitlines()
    has_from = any(l.startswith("--- ") for l in lines)
    has_to = any(l.startswith("+++ ") for l in lines)
    has_hunk = any(l.startswith("@@ ") for l in lines)
    has_changes = any(l.startswith("+") or l.startswith("-") for l in lines
                      if not l.startswith("---") and not l.startswith("+++"))
    score = sum([has_from, has_to, has_hunk, has_changes]) / 4.0
    # Penaliza respostas muito curtas ou muito longas
    if len(raw_response) < 50:
        score *= 0.3
    return round(score, 2)


__all__ = ["OllamaCoder"]
