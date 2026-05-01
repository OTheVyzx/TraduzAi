"""Critics locais do Lab — 100% rule-based, sem LLM, sem custo.

Cada critic recebe os artefatos de um capitulo (project.json ja carregado,
caminhos para source/reference/output) e devolve list[Finding] estruturado.

Uso:
    from lab.critics import run_all_critics
    findings = run_all_critics(chapter_artifact)
"""
from __future__ import annotations

from typing import Iterable

from lab.critics.base import Critic, Finding
from lab.critics.inpaint_critic import InpaintCritic
from lab.critics.ocr_critic import OcrCritic
from lab.critics.translation_critic import TranslationCritic
from lab.critics.typeset_critic import TypesetCritic

REGISTRY: list[Critic] = [
    OcrCritic(),
    TranslationCritic(),
    TypesetCritic(),
    InpaintCritic(),
]


def run_all_critics(chapter_artifact: dict) -> list[Finding]:
    """Executa todos os critics registrados sobre o artefato de um capitulo."""
    findings: list[Finding] = []
    for critic in REGISTRY:
        try:
            findings.extend(critic.analyze(chapter_artifact))
        except Exception as exc:  # pragma: no cover - critic nunca deve derrubar o Lab
            findings.append(
                Finding(
                    critic_id=critic.critic_id,
                    chapter_number=int(chapter_artifact.get("chapter_number", 0)),
                    page_index=-1,
                    bbox=None,
                    issue_type="critic_crashed",
                    severity="warning",
                    evidence={"exception": repr(exc)},
                    suggested_fix=f"Critic {critic.critic_id} lancou excecao; verifique logs.",
                )
            )
    return findings


def run_all_critics_over_chapters(chapter_artifacts: Iterable[dict]) -> list[Finding]:
    """Agrega findings de multiplos capitulos."""
    all_findings: list[Finding] = []
    for artifact in chapter_artifacts:
        all_findings.extend(run_all_critics(artifact))
    return all_findings


__all__ = [
    "Critic",
    "Finding",
    "REGISTRY",
    "run_all_critics",
    "run_all_critics_over_chapters",
]
