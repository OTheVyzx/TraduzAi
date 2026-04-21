"""ClaudeReviewerAgent — revisores de proposta usando Claude Code CLI.

Cada reviewer_id tenta primeiro o prompt especialista roteado pelo domínio real
do arquivo alvo. Se não houver especialista compatível, cai para prompts
genéricos de Rust/React/Tauri/Python conforme o reviewer.

Auth: o CLI `claude` usa OAuth da assinatura (via `claude setup-token`) ou
API key — o mesmo que o usuario ja configurou. Se o CLI nao estiver disponivel,
cai silenciosamente para a implementacao rule-based original.
"""
from __future__ import annotations

import re
from pathlib import Path

from lab.agents.base import (
    REVIEWER_DOMAIN_HINTS,
    get_agent_prompt,
    is_claude_available,
    read_file_slice,
)
from lab.agents.claude_cli_runner import invoke_claude_cli

MODEL = "claude-sonnet-4-5"
MAX_FILE_LINES = 200
MAX_TOKENS = 1024

# System prompts por reviewer_id (quando não existe agent especialista)
_INTEGRATION_ARCHITECT_SYSTEM = """\
Você é o Integration Architect do TraduzAi Lab. Revise propostas de melhoria e decida \
se estão prontas para integração. Avalie: risco de regressão, cobertura de testes, \
impacto em produção e consistência com a arquitetura existente.
Seja direto. Responda EXATAMENTE no formato solicitado (VERDICT + FINDING_N).
"""

_RUST_REVIEWER_SYSTEM = """\
Você é um engenheiro Rust sênior revisando código para o projeto TraduzAi (Tauri v2).
Avalie: segurança de tipos, gestão de erros, serialização serde, uso correto de async/tokio.
Responda EXATAMENTE no formato solicitado (VERDICT + FINDING_N).
"""

_REACT_REVIEWER_SYSTEM = """\
Você é um engenheiro React/TypeScript sênior revisando código para o projeto TraduzAi.
Avalie: tipagem correta, hooks React, chamadas invoke() Tauri, acessibilidade básica.
Responda EXATAMENTE no formato solicitado (VERDICT + FINDING_N).
"""

_TAURI_BOUNDARY_SYSTEM = """\
Você é um engenheiro sênior da fronteira Tauri/IPC do TraduzAi.
Avalie contratos entre React, bindings invoke(), comandos Rust, eventos e serde.
Responda EXATAMENTE no formato solicitado (VERDICT + FINDING_N).
"""

_GENERIC_PYTHON_SYSTEM = """\
Você é um engenheiro Python sênior revisando código para o projeto TraduzAi — \
uma aplicação de tradução automática de mangá com pipeline PaddleOCR + FT2Font.
Responda EXATAMENTE no formato solicitado (VERDICT + FINDING_N).
"""


class ClaudeReviewerAgent:
    """Executa revisões de proposta via Claude API com contexto especializado."""

    def review(
        self,
        proposal: dict,
        aggregate_benchmark: dict,
        reviewer_id: str,
        repo_root: Path | None = None,
    ) -> tuple[str, dict]:
        """Retorna (verdict, {"findings": [...]}).

        verdict: "approve" | "request_changes" | "needs_benchmark_focus"
        """
        if not is_claude_available():
            return _rule_based_review(proposal, aggregate_benchmark, reviewer_id)

        try:
            return self._claude_review(
                proposal, aggregate_benchmark, reviewer_id, repo_root
            )
        except Exception as exc:
            # Fallback gracioso — nunca deixa o runner travar
            verdict, payload = _rule_based_review(
                proposal, aggregate_benchmark, reviewer_id
            )
            payload["findings"].append(
                {
                    "title": f"Claude review falhou ({reviewer_id})",
                    "body": str(exc)[:300],
                    "severity": "info",
                    "file_path": "",
                }
            )
            return verdict, payload

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _claude_review(
        self,
        proposal: dict,
        aggregate_benchmark: dict,
        reviewer_id: str,
        repo_root: Path | None,
    ) -> tuple[str, dict]:
        target_file = str(proposal.get("target_file", ""))
        touched_domains = list(proposal.get("touched_domains") or [])

        # Escolhe o system prompt
        system = self._pick_system(reviewer_id, target_file, touched_domains)

        # Trecho do arquivo alvo (apenas para reviewers técnicos, não para architect)
        file_snippet = ""
        if reviewer_id != "integration_architect" and repo_root and target_file:
            p = repo_root / target_file
            if p.exists():
                file_snippet = read_file_slice(p, MAX_FILE_LINES)

        user_msg = _build_review_prompt(
            proposal, aggregate_benchmark, reviewer_id, file_snippet
        )

        text, err = invoke_claude_cli(
            prompt=user_msg,
            system=system,
            model=MODEL,
            cwd=repo_root,
        )
        if err:
            raise RuntimeError(err)

        return _parse_review_response(text, proposal, reviewer_id)

    def _pick_system(
        self, reviewer_id: str, target_file: str, touched_domains: list[str]
    ) -> str:
        if reviewer_id == "integration_architect":
            return _INTEGRATION_ARCHITECT_SYSTEM

        hinted_domains = list(touched_domains or [])
        hint = REVIEWER_DOMAIN_HINTS.get(reviewer_id)
        if hint:
            hinted_domains.append(hint)

        agent = get_agent_prompt(target_file, hinted_domains)
        if agent:
            return agent

        if reviewer_id == "rust_senior_reviewer":
            return _RUST_REVIEWER_SYSTEM
        if reviewer_id == "react_ts_senior_reviewer":
            return _REACT_REVIEWER_SYSTEM
        if reviewer_id == "tauri_boundary_reviewer":
            return _TAURI_BOUNDARY_SYSTEM

        return _GENERIC_PYTHON_SYSTEM


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------

def _build_review_prompt(
    proposal: dict,
    aggregate_benchmark: dict,
    reviewer_id: str,
    file_snippet: str,
) -> str:
    metrics = aggregate_benchmark.get("metrics") or {}
    score_after = float(aggregate_benchmark.get("score_after", 0))
    green = bool(aggregate_benchmark.get("green", False))

    metrics_text = (
        "\n".join(f"- {k}: {v:.1f}" for k, v in metrics.items())
        if metrics
        else "(sem métricas)"
    )

    file_section = ""
    if file_snippet:
        file_section = (
            f"\n\n## Trecho de `{proposal.get('target_file', '')}`\n"
            f"```python\n{file_snippet[:2000]}\n```"
        )

    return f"""\
## Proposta para revisão: {proposal.get('title', '')}

**Reviewer:** `{reviewer_id}`
**Issue type:** `{proposal.get('issue_type', '')}`
**Change kind:** `{proposal.get('change_kind', '')}`
**Risco:** `{proposal.get('risk', '')}`
**Arquivo alvo:** `{proposal.get('target_file', '')}`

**Motivação:**
{proposal.get('motivation', '')}

## Benchmark atual
- Score final: {score_after:.1f}
- Aprovado: {'Sim ✓' if green else 'Não ✗'}
{metrics_text}{file_section}

## Tarefa

Revise esta proposta e responda EXATAMENTE neste formato (sem texto extra antes do VERDICT):

VERDICT: <approve|request_changes|needs_benchmark_focus>
FINDING_1: <severity>|<título curto>|<descrição concreta, max 200 chars>
FINDING_2: <severity>|<título curto>|<descrição concreta, max 200 chars>

Use no máximo 3 FINDINGs. severity: info | warning | error.
Aponte código específico quando possível. Seja direto.
"""


# ------------------------------------------------------------------
# Response parser
# ------------------------------------------------------------------

def _parse_review_response(
    raw: str,
    proposal: dict,
    reviewer_id: str,
) -> tuple[str, dict]:
    verdict = "approve"
    m = re.search(r"VERDICT:\s*(\w+)", raw, re.IGNORECASE)
    if m:
        v = m.group(1).lower()
        if v in ("approve", "request_changes", "needs_benchmark_focus"):
            verdict = v

    findings: list[dict] = []
    for match in re.finditer(
        r"FINDING_\d+:\s*(\w+)\|([^|\n]+)\|(.+)", raw, re.IGNORECASE
    ):
        sev = match.group(1).strip().lower()
        title = match.group(2).strip()
        body = match.group(3).strip()
        findings.append(
            {
                "title": title[:120],
                "body": body[:400],
                "severity": sev if sev in ("info", "warning", "error") else "info",
                "file_path": str(proposal.get("target_file", "")),
            }
        )

    if not findings:
        # Fallback: usa o texto bruto como finding
        findings.append(
            {
                "title": f"Revisão Claude — {reviewer_id}",
                "body": raw[:400],
                "severity": "info",
                "file_path": str(proposal.get("target_file", "")),
            }
        )

    return verdict, {"findings": findings}


# ------------------------------------------------------------------
# Rule-based fallback (original logic de runner.py)
# ------------------------------------------------------------------

def _rule_based_review(
    proposal: dict,
    aggregate_benchmark: dict,
    reviewer_id: str,
) -> tuple[str, dict]:
    green = bool(aggregate_benchmark.get("green", False))
    metrics = aggregate_benchmark.get("metrics") or {}
    weakest = (
        min(metrics.items(), key=lambda x: float(x[1]))
        if metrics
        else ("readability", 0.0)
    )
    summary = aggregate_benchmark.get("summary", "Benchmark consolidado.")

    findings: list[dict] = [
        {
            "title": f"{reviewer_id} revisou o lote {proposal.get('batch_id', '')}",
            "body": summary,
            "severity": "info" if green else "warning",
            "file_path": "pipeline/**" if reviewer_id != "integration_architect" else "",
        }
    ]

    if reviewer_id == "integration_architect":
        findings.append(
            {
                "title": "Gate de integração",
                "body": (
                    "A rodada está pronta para promoção manual."
                    if green
                    else f"Antes de promover, precisamos subir {weakest[0]}."
                ),
                "severity": "info" if green else "warning",
                "file_path": "",
            }
        )
        verdict = "approve" if green else "request_changes"
    elif reviewer_id == "tauri_boundary_reviewer":
        verdict = "approve" if green else "needs_benchmark_focus"
    else:
        verdict = "approve" if green else "request_changes"

    return verdict, {"findings": findings}


__all__ = ["ClaudeReviewerAgent"]
